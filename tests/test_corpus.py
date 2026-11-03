from pathlib import Path
import pytest
from src.corpus import int_to_roman, load_chapters, roman_to_int

SAMPLE = """\
The Project Gutenberg eBook of Don Quijote

Licencia y texto legal que no debe aparecer en ningún capítulo.

*** START OF THE PROJECT GUTENBERG EBOOK DON QUIJOTE ***

Tasa y prólogo de la primera parte: tampoco deben aparecer.

Capítulo primero. Que trata de la condición
y ejercicio del famoso hidalgo

En un lugar de la Mancha, de cuyo nombre no quiero
acordarme, no ha mucho tiempo que vivía un hidalgo.

Segundo párrafo del capítulo uno.

Capítulo II. Que trata de la primera salida

Salió, pues, don Quijote al campo.
capítulo II. esto va en minúscula y no es un título.
Mira el Capítulo de los versos, que tampoco es un título.

Capítulo VIII. Del buen suceso que el valeroso don Quijote tuvo

Aquí está la aventura de los molinos.

Segunda parte del ingenioso caballero don Quijote de la Mancha

Tasa, dedicatoria y prólogo de la segunda parte: no son del capítulo VIII.

Capítulo Primero. De lo que el cura y el barbero pasaron con don Quijote

Texto del primer capítulo de la segunda parte.

Capítulo II. Que trata de la notable pendencia

Texto del segundo capítulo de la segunda parte.

*** END OF THE PROJECT GUTENBERG EBOOK DON QUIJOTE ***

Pie de licencia de Gutenberg que no debe aparecer.
"""

@pytest.fixture
def chapters(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text(SAMPLE, encoding="utf-8")
    return load_chapters(path)

def by_id(chapters, chapter_id):
    return next(c for c in chapters if c["id"] == chapter_id)

def test_roman_numerals_known_values():
    cases = {1: "I", 4: "IV", 9: "IX", 14: "XIV", 40: "XL", 49: "XLIX", 74: "LXXIV"}
    for number, roman in cases.items():
        assert int_to_roman(number) == roman
        assert roman_to_int(roman) == number

def test_roman_numerals_roundtrip():
    for n in range(1, 127):
        assert roman_to_int(int_to_roman(n)) == n

def test_chapter_count_and_parts(chapters):
    assert len(chapters) == 5
    assert [c["part"] for c in chapters] == [1, 1, 1, 2, 2]
    assert [c["number"] for c in chapters] == [1, 2, 8, 1, 2]

def test_ids_and_refs(chapters):
    assert [c["id"] for c in chapters] == [
        "p1-c01", "p1-c02", "p1-c08", "p2-c01", "p2-c02"
    ]
    assert by_id(chapters, "p1-c08")["ref"] == "Parte I, capítulo VIII"
    assert by_id(chapters, "p2-c01")["ref"] == "Parte II, capítulo I"

def test_primero_is_recognised_in_both_spellings(chapters):
    # "primero" en minúscula (Parte I) y "Primero" con mayúscula (Parte II)
    assert by_id(chapters, "p1-c01")["number"] == 1
    assert by_id(chapters, "p2-c01")["number"] == 1

def test_multiline_title_is_joined(chapters):
    title = by_id(chapters, "p1-c01")["title"]
    assert title == "Que trata de la condición y ejercicio del famoso hidalgo"

def test_header_and_footer_are_removed(chapters):
    everything = " ".join(c["text"] + " " + c["title"] for c in chapters)
    assert "Licencia y texto legal" not in everything
    assert "Pie de licencia" not in everything
    assert "Project Gutenberg" not in everything
    assert "Tasa y prólogo de la primera parte" not in everything

def test_part_two_preliminaries_do_not_leak_into_last_chapter_of_part_one(chapters):
    text = by_id(chapters, "p1-c08")["text"]
    assert "molinos" in text
    assert "Tasa, dedicatoria" not in text

def test_paragraphs_are_kept_and_broken_lines_joined(chapters):
    text = by_id(chapters, "p1-c01")["text"]
    paragraphs = text.split("\n\n")
    assert len(paragraphs) == 2
    assert "\n" not in paragraphs[0]
    assert paragraphs[0].startswith("En un lugar de la Mancha, de cuyo nombre no quiero acordarme")
    assert paragraphs[1] == "Segundo párrafo del capítulo uno."

def test_chapter_pattern_is_case_sensitive_and_anchored(chapters):
    text = by_id(chapters, "p1-c02")["text"]
    assert "capítulo II. esto va en minúscula" in text
    assert "Mira el Capítulo de los versos" in text
    assert len(chapters) == 5  # ninguna de esas líneas creó un capítulo nuevo

def test_crlf_and_bom_are_handled(tmp_path):
    path = tmp_path / "crlf.txt"
    data = SAMPLE.replace("\n", "\r\n").encode("utf-8")
    path.write_bytes(b"\xef\xbb\xbf" + data)
    result = load_chapters(path)
    assert [c["id"] for c in result] == [
        "p1-c01", "p1-c02", "p1-c08", "p2-c01", "p2-c02"
    ]
    assert "\r" not in result[0]["text"]
# Solo se ejecuta si tienes el libro descargado (data/ no se sube a git).
REAL_BOOK = Path("data/quijote.txt")

@pytest.mark.skipif(not REAL_BOOK.exists(), reason="data/quijote.txt no está en este PC")
def test_real_book_has_expected_structure():
    real = load_chapters(REAL_BOOK)
    assert len(real) == 126
    assert sum(1 for c in real if c["part"] == 1) == 52
    assert sum(1 for c in real if c["part"] == 2) == 74
    ids = [c["id"] for c in real]
    assert len(ids) == len(set(ids))
    assert all(c["text"] for c in real)