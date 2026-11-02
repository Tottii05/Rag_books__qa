import argparse
import json
import re
import sys
import time
import unicodedata

from src.llm import LLMError
from src.rag import ask

sys.stdout.reconfigure(encoding="utf-8")

# Amplía esta lista tras revisar las respuestas reales
REFUSAL_MARKERS = [
    "no aparece", "no he encontrado", "no se menciona", "no se mencionan",
    "no contienen", "no contiene", "no hay información", "no puedo responder",
    "no se puede responder", "no se encuentra",
]
PAUSE = 3          # segundos entre preguntas
MAX_RETRIES = 3
OUT_PATH = "eval/results_answers.json"

def ask_with_retry(question, top_k):
    for attempt in range(MAX_RETRIES):
        try:
            return ask(question, top_k=top_k, translate=True)
        except LLMError as exc:
            wait = 10 * (attempt + 1)
            print(f"  LLMError ({exc}); reintento {attempt + 1}/{MAX_RETRIES} en {wait}s")
            time.sleep(wait)
    return None  # agotados los reintentos: pregunta sin respuesta

def normalize(text):
    text = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"\s+", " ", text)

def is_refusal(answer):
    text = normalize(answer)
    return any(marker in text for marker in REFUSAL_MARKERS)

def cites_expected(answer, expected_refs):
    """True si la respuesta cita alguna referencia esperada.

    El lookahead evita que 'capítulo VI' case dentro de 'capítulo VIII'.
    """
    text = normalize(answer)
    for ref in expected_refs:
        pattern = re.escape(normalize(ref)) + r"(?![a-z])"
        if re.search(pattern, text):
            return True
    return False

def has_keywords(answer, keywords):
    """True si aparece al menos una palabra clave."""
    text = normalize(answer)
    return any(normalize(kw) in text for kw in keywords)

def rate(flags):
    return f"{sum(flags)}/{len(flags)}" if flags else "n/a"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="solo las N primeras preguntas (para probar)")
    parser.add_argument("--top-k", type=int, default=5,
                        help="fragmentos que recibe el LLM")
    args = parser.parse_args()

    with open("eval/questions.json", encoding="utf-8") as f:
        questions = json.load(f)
    if args.limit:
        questions = questions[: args.limit]

    rows = []
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q['id']}: {q['question']}")
        result = ask_with_retry(q["question"], args.top_k)
        row = {"id": q["id"], "type": q["type"], "lang": q["lang"],
               "question": q["question"], "answer": None, "refusal": None,
               "cited": None, "kw_hit": None}
        if result is not None:
            answer = result["answer"]
            row["answer"] = answer
            row["refusal"] = is_refusal(answer)
            if q["expected_refs"]:
                row["cited"] = cites_expected(answer, q["expected_refs"])
            if q["keywords"]:
                row["kw_hit"] = has_keywords(answer, q["keywords"])
        rows.append(row)
        time.sleep(PAUSE)

    # k=5 guarda en results_answers.json; otros valores, en un fichero aparte
    out_path = OUT_PATH if args.top_k == 5 else f"eval/results_answers_k{args.top_k}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"top_k": args.top_k, "rows": rows}, f, ensure_ascii=False, indent=2)

    answered = [r for r in rows if r["answer"] is not None]
    oos = [r for r in answered if r["type"] == "out_of_scope"]
    normal = [r for r in answered if r["type"] != "out_of_scope"]

    print(f"\n=== Métricas (top_k={args.top_k}) ===")
    print(f"Sin respuesta (límite/errores): {len(rows) - len(answered)}")
    print(f"Rechazo correcto en out_of_scope: {rate([r['refusal'] for r in oos])}")
    print(f"Falsos rechazos en el resto:      {rate([r['refusal'] for r in normal])}")
    print(f"Cita un capítulo esperado:        {rate([r['cited'] for r in normal if r['cited'] is not None])}")
    print(f"Contiene palabra clave:           {rate([r['kw_hit'] for r in normal if r['kw_hit'] is not None])}")

    print("\n=== Respuestas out_of_scope (revísalas a ojo) ===")
    for r in oos:
        print(f"\n[{r['id']}] {r['question']}\n-> refusal={r['refusal']}\n{r['answer']}")
    print(f"\nGuardado en {out_path}")


if __name__ == "__main__":
    main()