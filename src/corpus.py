"""Carga y limpieza de «Don Quijote» (Project Gutenberg #2000).

Convierte el .txt en una lista de capítulos con metadatos (parte, número,
título, referencia), para que cada respuesta del RAG pueda citar
«Parte I, capítulo VIII».

Uso desde la raíz del repo:
    python -m src.corpus                      # usa data/quijote.txt
    python -m src.corpus ruta/al/archivo.txt
"""
import argparse
import re
import sys
from pathlib import Path

START_MARK = "*** START OF THE PROJECT GUTENBERG EBOOK"
END_MARK = "*** END OF THE PROJECT GUTENBERG EBOOK"

# "Capítulo primero. Título..." o "Capítulo II. Título..."
# Sin IGNORECASE a propósito: evita falsos positivos en medio del texto.
CHAPTER_RE = re.compile(r"^Capítulo\s+([Pp]rimero|[IVXLC]+)\.\s*(.*)$")
PART2_RE = re.compile(r"^Segunda parte del ingenioso caballero", re.IGNORECASE)

EXPECTED_CHAPTERS = {1: 52, 2: 74}  # capítulos de cada parte del Quijote

_ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}
_ROMAN_PAIRS = [(100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
                (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]


def roman_to_int(s: str) -> int:
    total = 0
    for i, ch in enumerate(s):
        value = _ROMAN_VALUES[ch]
        nxt = _ROMAN_VALUES[s[i + 1]] if i + 1 < len(s) else 0
        total += -value if value < nxt else value
    return total


def int_to_roman(n: int) -> str:
    out = ""
    for value, symbol in _ROMAN_PAIRS:
        while n >= value:
            out += symbol
            n -= value
    return out


def _clean_body(lines: list[str]) -> str:
    """Une las líneas partidas de cada párrafo y conserva los párrafos."""
    text = "\n".join(lines)
    paragraphs = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", text)]
    return "\n\n".join(p for p in paragraphs if p)


def load_chapters(path: str | Path) -> list[dict]:
    """Devuelve los capítulos como dicts: id, part, number, title, ref, text.

    Se descartan la cabecera/pie de licencia de Gutenberg, el índice inicial y
    los preliminares (tasa, dedicatorias, prólogos, versos) de ambas partes.
    """
    raw = Path(path).read_text(encoding="utf-8-sig").replace("\r\n", "\n")

    if START_MARK in raw:
        raw = raw.split(START_MARK, 1)[1].split("\n", 1)[1]
    if END_MARK in raw:
        raw = raw.split(END_MARK, 1)[0]

    lines = raw.split("\n")
    chapters: list[dict] = []
    current: dict | None = None
    part, last_number = 1, 0
    i = 0

    while i < len(lines):
        stripped = lines[i].strip()
        match = CHAPTER_RE.match(stripped)

        if match:
            number = 1 if match.group(1).lower() == "primero" else roman_to_int(match.group(1))
            # El título puede continuar en las líneas siguientes hasta una en blanco.
            title_parts = [match.group(2).strip()]
            j = i + 1
            while j < len(lines) and lines[j].strip():
                title_parts.append(lines[j].strip())
                j += 1

            if number <= last_number:  # la numeración se reinicia: empieza la Parte II
                part += 1
            last_number = number

            current = {
                "id": f"p{part}-c{number:02d}",
                "part": part,
                "number": number,
                "title": " ".join(title_parts).strip(),
                "ref": f"Parte {int_to_roman(part)}, capítulo {int_to_roman(number)}",
                "_lines": [],
            }
            chapters.append(current)
            i = j
            continue

        if PART2_RE.match(stripped) and chapters:
            current = None  # lo que sigue (tasa, prólogo...) no es del último capítulo
        elif current is not None:
            current["_lines"].append(lines[i])
        i += 1

    for chapter in chapters:
        chapter["text"] = _clean_body(chapter.pop("_lines"))
    return chapters


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # acentos correctos en Windows
    parser = argparse.ArgumentParser(description="Comprueba la carga del corpus.")
    parser.add_argument("path", nargs="?", default="data/quijote.txt")
    args = parser.parse_args()

    chapters = load_chapters(args.path)
    print(f"Capítulos cargados: {len(chapters)}")
    for part, expected in EXPECTED_CHAPTERS.items():
        found = sum(1 for c in chapters if c["part"] == part)
        status = "OK" if found == expected else f"AVISO (esperados {expected})"
        print(f"  Parte {int_to_roman(part)}: {found} capítulos -> {status}")

    if chapters:
        lengths = [len(c["text"]) for c in chapters]
        print(f"Caracteres por capítulo: mín {min(lengths)}, "
              f"máx {max(lengths)}, total {sum(lengths)}")
        first = chapters[0]
        print(f"\n[{first['ref']}] {first['title']}")
        print(first["text"][:300] + "...")


if __name__ == "__main__":
    main()