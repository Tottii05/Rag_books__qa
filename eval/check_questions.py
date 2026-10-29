"""Valida eval/questions.json contra el corpus real.

Comprueba que cada capítulo esperado existe y enseña su título, para que puedas
confirmar a ojo que la referencia es la correcta. También valida la estructura
del archivo (ids únicos, tipos, idiomas...).

Uso desde la raíz del repo, con el entorno activo:
    python -m eval.check_questions
"""
import json
import sys
from collections import Counter
from pathlib import Path

from src.corpus import load_chapters

QUESTIONS_PATH = Path("eval/questions.json")
CORPUS_PATH = "data/quijote.txt"
TYPES = {"answerable", "out_of_scope"}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # acentos correctos en Windows
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    titles = {c["ref"]: c["title"] for c in load_chapters(CORPUS_PATH)}

    problems: list[str] = []
    ids = Counter(q["id"] for q in questions)
    problems += [f"id repetido: {i}" for i, n in ids.items() if n > 1]

    for q in questions:
        print(f"\n{q['id']} [{q['type']}/{q['lang']}] {q['question']}")
        if q["type"] not in TYPES:
            problems.append(f"{q['id']}: tipo desconocido '{q['type']}'")
        if q["type"] == "answerable" and not q["expected_refs"]:
            problems.append(f"{q['id']}: es 'answerable' pero no tiene expected_refs")
        if q["type"] == "out_of_scope" and q["expected_refs"]:
            problems.append(f"{q['id']}: es 'out_of_scope' pero tiene expected_refs")
        if q["keywords"] and q["lang"] != "es" and q["type"] == "out_of_scope":
            problems.append(f"{q['id']}: out_of_scope no debería tener keywords")

        for ref in q["expected_refs"]:
            title = titles.get(ref)
            if title is None:
                print(f"   ✗ {ref}  -> NO EXISTE en el corpus")
                problems.append(f"{q['id']}: la referencia '{ref}' no existe")
            else:
                print(f"   ✓ {ref}: {title[:95]}")

    summary = Counter((q["type"], q["lang"]) for q in questions)
    print(f"\n{len(questions)} preguntas: {dict(summary)}")
    if problems:
        print(f"\n{len(problems)} PROBLEMA(S):")
        for problem in problems:
            print("  -", problem)
        sys.exit(1)
    print("Estructura y referencias correctas. Ahora revisa a ojo que cada título "
          "encaja con su pregunta.")


if __name__ == "__main__":
    main()