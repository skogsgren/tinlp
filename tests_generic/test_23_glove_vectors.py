from pathlib import Path

from tinlp.utils.text import search_vectors


VECTORS = Path("./data/sm.vector.gz")
GOLDEN = [
    "wine",
    "water",
    "2",
    "again",
    "together",
    "hops",
    "off",
    "wines",
    "beer",
    "raspberries",
]


def test_vector_similarity_fn():
    for k in search_vectors(VECTORS, "wine"):
        assert k in GOLDEN
    assert not search_vectors(VECTORS, "cornflakes")
