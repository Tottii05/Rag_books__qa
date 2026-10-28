import argparse
import os
import sys
import time

from dotenv import load_dotenv
from pinecone import Pinecone
from src.chunking import chunk_corpus
from src.corpus import load_chapters

MODEL = "llama-text-embed-v2"
TEXT_FIELD = "chunk_text"          # campo que Pinecone convierte en vector
DEFAULT_NAMESPACE = "quijote-es"
BATCH_SIZE = 96                    # máximo de registros con texto por petición
TOKENS_PER_MINUTE = 200_000
CHARS_PER_TOKEN = 4                # estimación aproximada para español

def estimate_tokens(texts: list[str]) -> int:
    return sum(len(t) for t in texts) // CHARS_PER_TOKEN

def to_record(chunk: dict) -> dict:
    """Convierte un fragmento al formato de Pinecone (metadatos planos)."""
    return {
        "_id": chunk["id"],
        TEXT_FIELD: chunk["text"],
        "ref": chunk["ref"],
        "title": chunk["title"],
        "part": chunk["part"],
        "chapter": chunk["chapter"],
    }

def get_index(pc: Pinecone, name: str):
    """Devuelve el índice, creándolo si no existe."""
    if not pc.has_index(name):
        print(f"Creando el índice '{name}' ({MODEL})...")
        created = pc.create_index_for_model(
            name=name,
            cloud="aws",
            region="us-east-1",
            embed={"model": MODEL, "field_map": {"text": TEXT_FIELD}},
        )
        host = created.host
    else:
        host = pc.describe_index(name).host
    return pc.Index(host=host)

def upsert_with_retry(index, namespace: str, records: list[dict], max_attempts: int = 6) -> None:
    for attempt in range(1, max_attempts + 1):
        try:
            index.upsert_records(namespace=namespace, records=records)
            return
        except Exception as exc:  # noqa: BLE001
            status = getattr(exc, "status_code", None)
            sin_remedio = status is not None and 400 <= status < 500 and status != 429
            if attempt == max_attempts or sin_remedio:
                raise
            wait = min(5 * 2 ** (attempt - 1), 60)
            print(f"  Error ({type(exc).__name__}: {str(exc)[:120]}). "
                  f"Reintento {attempt}/{max_attempts - 1} en {wait}s...")
            time.sleep(wait)

def upload(index, namespace: str, records: list[dict]) -> None:
    """Sube los registros en lotes, respetando el límite de tokens por minuto."""
    batches = [records[i:i + BATCH_SIZE] for i in range(0, len(records), BATCH_SIZE)]
    for number, batch in enumerate(batches, start=1):
        started = time.monotonic()
        upsert_with_retry(index, namespace, batch)
        print(f"  Lote {number}/{len(batches)} subido ({len(batch)} fragmentos)")
        if number < len(batches):
            tokens = estimate_tokens([r[TEXT_FIELD] for r in batch])
            pause = tokens * 60 / TOKENS_PER_MINUTE - (time.monotonic() - started)
            if pause > 0:
                time.sleep(pause)

def _get(obj, name: str):
    """Lee un campo tanto si el objeto es un diccionario como si tiene atributos."""
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)

def namespace_count(index, namespace: str) -> int:
    """Número de registros que Pinecone indica para un namespace."""
    stats = index.describe_index_stats()
    info = (_get(stats, "namespaces") or {}).get(namespace)
    return int(_get(info, "vector_count") or 0) if info is not None else 0

def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # acentos correctos en Windows
    parser = argparse.ArgumentParser(description="Sube el corpus a Pinecone.")
    parser.add_argument("path", nargs="?", default="data/quijote.txt")
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    parser.add_argument("--max-chapters", type=int, help="indexar solo los N primeros capítulos")
    parser.add_argument("--dry-run", action="store_true", help="calcular sin subir nada")
    parser.add_argument("--yes", action="store_true", help="no pedir confirmación")
    args = parser.parse_args()
    chapters = load_chapters(args.path)
    if args.max_chapters:
        chapters = chapters[:args.max_chapters]
    chunks = chunk_corpus(chapters)
    records = [to_record(c) for c in chunks]
    tokens = estimate_tokens([r[TEXT_FIELD] for r in records])

    print(f"Capítulos: {len(chapters)} | Fragmentos: {len(records)} | "
          f"Tokens estimados: ~{tokens:,} (cuota mensual gratuita: 5.000.000)")
    print(f"Namespace: '{args.namespace}'")
    if args.dry_run:
        print("Modo --dry-run: no se ha subido nada.")
        return
    load_dotenv()
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX", "book-qa-index")
    if not api_key:
        sys.exit("Falta PINECONE_API_KEY en el archivo .env")
    if not args.yes and input("¿Subir a Pinecone? [s/N] ").strip().lower() != "s":
        print("Cancelado.")
        return
    pc = Pinecone(api_key=api_key)
    index = get_index(pc, index_name)
    print(f"Subiendo a '{index_name}'...")
    started = time.monotonic()
    upload(index, args.namespace, records)
    # Pinecone tarda unos segundos en reflejar los registros nuevos.
    count = 0
    for _ in range(6):
        time.sleep(5)
        count = namespace_count(index, args.namespace)
        if count >= len(records):
            break
    state = "OK" if count >= len(records) else "AVISO: aún no coincide, espera y revisa en la consola"
    print(f"Registros en '{args.namespace}': {count} de {len(records)} -> {state}")
    print(f"Tiempo total: {time.monotonic() - started:.0f} s")


if __name__ == "__main__":
    main()