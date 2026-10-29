import argparse
import sys
from src.llm import LLMError, answer, to_spanish
from src.retrieval import search

def ask(question: str, top_k: int = 5, translate: bool = True) -> dict:
    """Traduce la pregunta, busca fragmentos y redacta la respuesta."""
    if translate:
        search_query = to_spanish(question)
    else:
        search_query = question
    hits = search(search_query, top_k)
    reply = answer(question, hits)  # la pregunta ORIGINAL: así responde en su idioma
    return {"answer": reply, "hits": hits, "search_query": search_query}

def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # acentos correctos en Windows
    parser = argparse.ArgumentParser(description="Pregunta al Quijote.")
    parser.add_argument("question", nargs="+", help="la pregunta")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    question = " ".join(args.question)
    try:
        result = ask(question, top_k=args.top_k)
    except LLMError as exc:
        sys.exit(f"Error: {exc}")

    print(result["answer"])
    print(f"\nBúsqueda realizada con: {result['search_query']}")
    print("Fragmentos consultados:",", ".join(dict.fromkeys(hit["ref"] for hit in result["hits"])))

if __name__ == "__main__":
    main()