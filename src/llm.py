import argparse
import os
import sys
from functools import lru_cache
import openai
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

SYSTEM_PROMPT = (
    "Eres un asistente experto en «Don Quijote de la Mancha». "
    "Responde ÚNICAMENTE con la información de los fragmentos que se te proporcionan. "
    "Si los fragmentos no contienen la respuesta, di claramente que no aparece en "
    "los fragmentos consultados; no inventes nada. "
    "Después de cada afirmación cita el capítulo entre corchetes, por ejemplo "
    "[Parte I, capítulo VIII]. "
    "Detecta el idioma de la pregunta y responde SIEMPRE en ese idioma "
    "(por ejemplo, si la pregunta está en inglés, responde en inglés), aunque "
    "los fragmentos estén en español. Sé claro y breve."
)

TRANSLATE_PROMPT = (
    "Traduce la pregunta del usuario al español. Si ya está en español, "
    "devuélvela exactamente igual. Responde SOLO con la pregunta, sin comillas "
    "ni explicaciones."
)

NO_HITS_MESSAGE = "No he encontrado fragmentos del libro relacionados con tu pregunta."

class LLMError(RuntimeError):
    """Error con un mensaje pensado para enseñárselo al usuario."""

@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    """Cliente para Groq, Ollama o cualquier API compatible (según el .env)."""
    return OpenAI(api_key=os.getenv("LLM_API_KEY"), base_url=os.getenv("LLM_BASE_URL"))

def build_messages(question: str, hits: list[dict]) -> list[dict]:
    """Construye los mensajes del chat: instrucciones + fragmentos + pregunta."""
    fragments = "\n\n".join(f"[{hit['ref']}]\n{hit['text']}" for hit in hits)
    user_content = (f"Fragmentos:\n\n{fragments}\n\nPregunta: {question}\n\n""Responde en el mismo idioma en que está escrita la pregunta.")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

def _chat(messages: list[dict], temperature: float = 0.2) -> str:
    """Envía unos mensajes ya construidos al LLM y devuelve su texto."""
    model = os.getenv("LLM_MODEL")
    if not model:
        raise LLMError("Falta LLM_MODEL en el archivo .env")

    try:
        response = get_client().chat.completions.create(
            model=model,
            messages=messages,  # ya vienen construidos: no se vuelven a construir
            temperature=temperature,
        )
    except openai.RateLimitError as exc:
        raise LLMError("Se ha alcanzado el límite de peticiones del LLM. "
                       "Espera un minuto e inténtalo de nuevo.") from exc
    except openai.AuthenticationError as exc:
        raise LLMError("La clave del LLM no es válida: revisa LLM_API_KEY en .env") from exc
    except openai.NotFoundError as exc:
        raise LLMError(f"El modelo '{model}' no existe en este servicio: "
                       "revisa LLM_MODEL en .env") from exc
    except openai.APIConnectionError as exc:
        raise LLMError("No se pudo conectar con el LLM: revisa LLM_BASE_URL "
                       "y tu conexión a internet.") from exc

    text = response.choices[0].message.content
    if not text:
        raise LLMError("El modelo devolvió una respuesta vacía. "
                       "Prueba con otro modelo cambiando LLM_MODEL en .env")
    return text.strip()

def answer(question: str, hits: list[dict]) -> str:
    """Devuelve la respuesta del LLM a partir de la pregunta y los fragmentos."""
    if not hits:  # sin fragmentos no tiene sentido gastar una llamada al LLM
        return NO_HITS_MESSAGE
    return _chat(build_messages(question, hits))

def to_spanish(question: str) -> str:
    """Traduce la pregunta al español para buscar en el libro.

    La traducción es una mejora opcional: si algo falla, se usa la pregunta original.
    """
    messages = [
        {"role": "system", "content": TRANSLATE_PROMPT},
        {"role": "user", "content": question},
    ]
    try:
        translated = _chat(messages, temperature=0)  # temperatura 0: traducción estable
    except LLMError:
        return question
    translated = translated.strip('"«» \n')  # quita comillas, espacios y saltos sobrantes
    return translated or question

def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # acentos correctos en Windows
    parser = argparse.ArgumentParser(description="Pregunta al Quijote.")
    parser.add_argument("question", nargs="+", help="la pregunta")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    # Importado aquí para que llm.py se pueda usar sin necesitar Pinecone.
    from src.retrieval import search

    question = " ".join(args.question)
    hits = search(question, top_k=args.top_k)
    try:
        print(answer(question, hits))
    except LLMError as exc:
        sys.exit(f"Error: {exc}")
    print("\nFragmentos consultados:", ", ".join(dict.fromkeys(h["ref"] for h in hits)))

if __name__ == "__main__":
    main()