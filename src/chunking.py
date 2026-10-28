import argparse
import re
import sys

from src.corpus import load_chapters

# Fin de frase: signo de cierre + espacio + inicio de frase nueva.
SENTENCE_END = re.compile(r'(?<=[.!?…»])\s+(?=[A-ZÁÉÍÓÚÑ¿¡—«"])')
# Fin de cláusula, para partir frases larguísimas.
CLAUSE_END = re.compile(r"(?<=[,;:])\s+")

def split_sentences(text: str) -> list[str]:
    """Divide un texto en frases."""
    text = text.replace("\n", " ")
    return [s.strip() for s in SENTENCE_END.split(text) if s.strip()]

def split_long_sentence(sentence: str, max_chars: int) -> list[str]:
    pieces: list[str] = []
    current = ""
    for clause in CLAUSE_END.split(sentence):
        candidate = f"{current} {clause}".strip()
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            pieces.append(current)
        current = clause
        while len(current) > max_chars:
            cut = current.rfind(" ", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            pieces.append(current[:cut].strip())
            current = current[cut:].strip()
    if current:
        pieces.append(current)
    return pieces


def make_chunk(chapter: dict, index: int, sentences: list[str]) -> dict:
    return {
        "id": f"{chapter['id']}-{index:03d}",
        "text": " ".join(sentences),
        "ref": chapter["ref"],
        "title": chapter["title"],
        "part": chapter["part"],
        "chapter": chapter["number"],
        "index": index,
    }

def chunk_chapter(chapter: dict, chunk_size: int = 1000, overlap: int = 200) -> list[dict]:
    max_sentence = int(chunk_size * 1.5)
    sentences: list[str] = []
    for sentence in split_sentences(chapter["text"]):
        if len(sentence) > max_sentence:
            sentences.extend(split_long_sentence(sentence, chunk_size))
        else:
            sentences.append(sentence)
    chunks: list[dict] = []
    current: list[str] = []  # frases del fragmento en construcción
    current_len = 0          # caracteres acumulados (contando los espacios)
    for sentence in sentences:
        added = len(sentence) + (1 if current else 0)
        if current and current_len + added > chunk_size:
            chunks.append(make_chunk(chapter, len(chunks), current))
            # recorriendo el fragmento de atrás hacia delante.
            tail: list[str] = []
            tail_len = 0
            for previous in reversed(current):
                extra = len(previous) + (1 if tail else 0)
                if tail_len + extra > overlap:
                    break
                tail.insert(0, previous)
                tail_len += extra
            current, current_len = tail, tail_len
            added = len(sentence) + (1 if current else 0)
        current.append(sentence)
        current_len += added
    if current:  # el último fragmento no se cierra solo
        chunks.append(make_chunk(chapter, len(chunks), current))
    return chunks

def chunk_corpus(chapters: list[dict], chunk_size: int = 1000, overlap: int = 200) -> list[dict]:
    """Trocea todos los capítulos y devuelve una única lista plana."""
    chunks: list[dict] = []
    for chapter in chapters:
        chunks.extend(chunk_chapter(chapter, chunk_size, overlap))
    return chunks

def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # acentos correctos en Windows
    parser = argparse.ArgumentParser(description="Trocea el corpus y muestra un resumen.")
    parser.add_argument("path", nargs="?", default="data/quijote.txt")
    chunks = chunk_corpus(load_chapters(parser.parse_args().path))
    sizes = [len(c["text"]) for c in chunks]
    print(f"{len(chunks)} fragmentos | media {sum(sizes) // len(sizes)} caracteres")
    
if __name__ == "__main__":
    main()