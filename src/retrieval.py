import argparse
import os
import sys
from functools import lru_cache

from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()  # carga las variables del archivo .env

NAMESPACE = "quijote-es"
FIELDS = ["chunk_text", "ref", "title"]

@lru_cache(maxsize=1)
def get_index():
    """Conecta con el índice una sola vez (lru_cache) y lo reutiliza."""
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("Falta PINECONE_API_KEY en el archivo .env")
    index_name = os.getenv("PINECONE_INDEX", "quijote-qa")
    pc = Pinecone(api_key=api_key)
    return pc.Index(host=pc.describe_index(index_name).host)

def search(question: str, top_k: int = 5, namespace: str = NAMESPACE) -> list[dict]:
    """Devuelve los top_k fragmentos más parecidos a la pregunta.
    Cada resultado es un diccionario: id, score, ref, title y text.
    """
    response = get_index().search(
        namespace=namespace,
        inputs={"text": question},
        top_k=top_k,
        fields=FIELDS,
    )
    return [
        {
            "id": hit.id_,
            "score": hit.score_,
            "ref": hit.fields.get("ref", ""),
            "title": hit.fields.get("title", ""),
            "text": hit.fields.get("chunk_text", ""),
        }
        for hit in response.result.hits
    ]

def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # acentos correctos en Windows
    parser = argparse.ArgumentParser(description="Busca fragmentos del Quijote.")
    parser.add_argument("question", nargs="+", help="la pregunta")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    hits = search(" ".join(args.question), top_k=args.top_k)
    for number, hit in enumerate(hits, start=1):
        print(f"{number}. [{hit['score']:.3f}] {hit['ref']} - {hit['title'][:70]}")
        print(f"   {hit['text'][:200]}...\n")

if __name__ == "__main__":
    main()