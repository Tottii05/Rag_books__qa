import logging
import streamlit as st
from src.llm import LLMError, answer
from src.retrieval import search

logger = logging.getLogger(__name__)
st.set_page_config(page_title="Pregunta al Quijote", page_icon="📖")

EXAMPLES = [
    "¿Qué le pasó a don Quijote con los molinos de viento?",
    "¿Cómo se llama el caballo de don Quijote?",
    "¿Qué es el yelmo de Mambrino?",
    "¿Qué ínsula gobierna Sancho Panza?",
    "¿Cómo muere don Quijote?",
]

# ---------- Barra lateral ----------
with st.sidebar:
    st.header("Ajustes")
    top_k = st.slider(
        "Fragmentos a consultar", min_value=3, max_value=8, value=5,
        help="Cuántos trozos del libro se le dan al modelo para responder.",
    )
    st.markdown("**Preguntas de ejemplo**")
    for example in EXAMPLES:
        if st.button(example, width="stretch"):
            st.session_state.pending_question = example
    st.divider()
    st.caption("RAG sobre el texto completo de *Don Quijote de la Mancha* (Project Gutenberg). Búsqueda semántica en Pinecone; las respuestas las redacta un LLM usando solo los fragmentos recuperados.")

# ---------- Cabecera ----------
st.title("Pregunta al Quijote")
st.caption("Cada respuesta cita el capítulo del que sale. Puedes preguntar en español, catalán o inglés.")

if "messages" not in st.session_state:
    st.session_state.messages = []

def render_sources(hits: list[dict]) -> None:
    """Muestra los fragmentos del libro que se usaron para responder."""
    with st.expander(f"Fragmentos consultados ({len(hits)})"):
        for hit in hits:
            st.markdown(f"**{hit['ref']}** · {hit['title']}  \n*similitud {hit['score']:.2f}*")
            st.write(hit["text"])
            st.divider()

# ---------- Historial ----------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            render_sources(message["sources"])

# ---------- Nueva pregunta ----------
question = st.chat_input("Pregunta algo sobre el Quijote...") or st.session_state.pop("pending_question", None)
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Buscando en el libro..."):
            try:
                hits = search(question, top_k=top_k)
                reply = answer(question, hits)
            except LLMError as exc:  # errores previstos, con mensaje amable
                hits, reply = [], f"{exc}"
            except Exception:  # noqa: BLE001 - cualquier otro fallo (red, Pinecone...)
                logger.exception("Error al responder")
                hits, reply = [], ("No he podido consultar el libro ahora mismo. Inténtalo de nuevo en un momento.")
        st.markdown(reply)
        if hits:
            render_sources(hits)
    st.session_state.messages.append({"role": "assistant", "content": reply, "sources": hits})