import re
from src.chunking import chunk_corpus

CHUNK_SIZE = 1000
OVERLAP = 200

def make_chapter(number, text, part=1):
    return {
        "id": f"p{part}-c{number}",
        "part": part,
        "number": number,
        "title": f"Título {number}",
        "ref": f"Parte I, capítulo {number}",
        "text": text,
    }

def numbered_sentences(prefix, n):
    """Frases cortas y distintas: 'ALFA frase número 1. ALFA frase número 2. ...'"""
    return " ".join(f"{prefix} frase número {i}." for i in range(1, n + 1))

def chunks_of(chunks, chapter):
    return [c for c in chunks if c["ref"] == chapter["ref"]]

def test_chunks_do_not_cross_chapters():
    ch1 = make_chapter(1, numbered_sentences("ALFA", 80))
    ch2 = make_chapter(2, numbered_sentences("BETA", 80))
    chunks = chunk_corpus([ch1, ch2], chunk_size=CHUNK_SIZE, overlap=OVERLAP)

    assert chunks, "debería haber fragmentos"
    for c in chunks:
        assert not ("ALFA" in c["text"] and "BETA" in c["text"])
        if "ALFA" in c["text"]:
            assert c["ref"] == ch1["ref"]
        if "BETA" in c["text"]:
            assert c["ref"] == ch2["ref"]

def test_chunk_size_is_respected():
    ch = make_chapter(1, numbered_sentences("ALFA", 200))
    chunks = chunk_corpus([ch], chunk_size=CHUNK_SIZE, overlap=OVERLAP)
    assert len(chunks) > 1
    # Margen holgado por si el código deja pasar una frase más del límite
    for c in chunks:
        assert len(c["text"]) <= CHUNK_SIZE + OVERLAP

def test_overlap_shares_a_sentence():
    ch = make_chapter(1, numbered_sentences("ALFA", 200))
    chunks = chunk_corpus([ch], chunk_size=CHUNK_SIZE, overlap=OVERLAP)
    assert len(chunks) > 1
    for current, following in zip(chunks, chunks[1:]):
        last_sentence = re.findall(r"ALFA frase número \d+\.", current["text"])[-1]
        assert last_sentence in following["text"]

def test_no_sentence_is_lost():
    n = 150
    ch = make_chapter(1, numbered_sentences("ALFA", n))
    chunks = chunk_corpus([ch], chunk_size=CHUNK_SIZE, overlap=OVERLAP)
    joined = " ".join(c["text"] for c in chunks)
    for i in range(1, n + 1):
        assert f"ALFA frase número {i}." in joined

def test_metadata_is_consistent():
    ch1 = make_chapter(1, numbered_sentences("ALFA", 80))
    ch2 = make_chapter(2, numbered_sentences("BETA", 80))
    chunks = chunk_corpus([ch1, ch2], chunk_size=CHUNK_SIZE, overlap=OVERLAP)
    ids = [c["id"] for c in chunks]
    assert len(ids) == len(set(ids)), "los ids deben ser únicos"
    for ch in (ch1, ch2):
        own = chunks_of(chunks, ch)
        assert own
        for c in own:
            assert c["title"] == ch["title"]
            assert c["part"] == ch["part"]
            assert c["chapter"] == ch["number"]
        indexes = [c["index"] for c in own]
        assert indexes == list(range(indexes[0], indexes[0] + len(indexes)))

def test_very_long_sentence_is_split():
    # Una sola "frase" de ~2000 caracteres con comas y un único punto final
    long_sentence = ", ".join(f"palabra{i}" for i in range(280)) + "."
    assert len(long_sentence) > 1900
    ch = make_chapter(1, long_sentence)
    chunks = chunk_corpus([ch], chunk_size=CHUNK_SIZE, overlap=OVERLAP)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c["text"]) <= 1500

def test_empty_chapter_gives_no_chunks():
    chunks = chunk_corpus([make_chapter(1, "")], chunk_size=CHUNK_SIZE, overlap=OVERLAP)
    assert chunks == []