import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from src.llm import to_spanish
from src.retrieval import search

QUESTIONS_PATH = Path("eval/questions.json")
RESULTS_PATH = Path("eval/results_retrieval.json")
MAX_K = 8
KS = (3, 5, 8)

def first_rank(hits: list[dict], expected_refs: list[str]) -> int | None:
    """Posición (1 = primero) del primer fragmento de un capítulo esperado, o None."""
    for position, hit in enumerate(hits, start=1):
        if hit["ref"] in expected_refs:
            return position
    return None

def hit_at(ranks: list, k: int) -> float:
    """Fracción de preguntas cuyo capítulo esperado está entre los k primeros."""
    return sum(1 for r in ranks if r is not None and r <= k) / len(ranks)

def mrr(ranks: list) -> float:
    """Rango recíproco medio: 1 si sale el 1º, 1/2 si el 2º... y 0 si no sale."""
    return sum(1 / r for r in ranks if r is not None) / len(ranks)

def evaluate(questions: list[dict]) -> list[dict]:
    """Una fila por (pregunta, modo) con la posición del capítulo esperado."""
    rows = []
    for q in questions:
        if q["type"] != "answerable":  # las de fuera del libro se miden en otro paso
            continue
        queries = [("original", q["question"])]
        if q["lang"] != "es":  # en español no tiene sentido traducir
            queries.append(("traducida", to_spanish(q["question"])))

        for mode, query in queries:
            hits = search(query, top_k=MAX_K)  # una búsqueda de 8 sirve para hit@3, @5 y @8
            rows.append({
                "id": q["id"], "lang": q["lang"], "mode": mode,
                "question": q["question"], "query": query,
                "rank": first_rank(hits, q["expected_refs"]),
            })
    return rows

def summarize(rows: list[dict]) -> dict:
    """Métricas por grupo (idioma, modo)."""
    groups = defaultdict(list)
    for row in rows:
        groups[(row["lang"], row["mode"])].append(row["rank"])
    return {
        group: {"n": len(ranks), **{f"hit@{k}": hit_at(ranks, k) for k in KS}, "mrr": mrr(ranks)}
        for group, ranks in groups.items()
    }

def format_table(summary: dict) -> str:
    lines = [f"{'grupo':<15}{'n':>3}  " + " ".join(f"{'hit@' + str(k):>6}" for k in KS) + f" {'MRR':>6}"]
    for lang, mode in sorted(summary, key=lambda key: (key[0] != "es", key)):  # español primero
        s = summary[(lang, mode)]
        metrics = " ".join(f"{s[f'hit@{k}']:>6.2f}" for k in KS)
        lines.append(f"{lang + ' ' + mode:<15}{s['n']:>3}  {metrics} {s['mrr']:>6.2f}")
    return "\n".join(lines)

def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # acentos correctos en Windows
    parser = argparse.ArgumentParser(description="Evalúa la recuperación del RAG.")
    parser.add_argument("--output", type=Path, default=RESULTS_PATH, help="dónde guardar los resultados en JSON")
    args = parser.parse_args()
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    rows = evaluate(questions)
    summary = summarize(rows)
    print(format_table(summary))
    misses = [r for r in rows if r["rank"] is None]
    if misses:
        print(f"\nSin el capítulo esperado en el top {MAX_K} ({len(misses)}):")
        for r in misses:
            print(f"  {r['id']} ({r['lang']}, {r['mode']}): {r['question']}")
    # Si la traducción no cambió la pregunta, el modo "traducida" no mide nada.
    unchanged = [r["id"] for r in rows if r["mode"] == "traducida" and r["query"] == r["question"]]
    if unchanged:
        print(f"\nAVISO: la traducción no cambió la pregunta en {unchanged}. "
              "Puede haber fallado (límite de peticiones del LLM, clave...).")
    # JSON no admite tuplas como clave: (lang, mode) pasa a "lang/mode".
    results = {
        "max_k": MAX_K,
        "summary": {f"{lang}/{mode}": metrics for (lang, mode), metrics in summary.items()},
        "rows": rows,
    }
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nResultados guardados en {args.output}")

if __name__ == "__main__":
    main()