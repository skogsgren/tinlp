from pathlib import Path

from tinlp.utils.text import search_vectors


VECTORS = Path("./data/glove-en.vec.gz")
VECTORS = Path("./test.vector")


def test_vector_similarity_fn():
    print(search_vectors(VECTORS, "friday"))
    # print(search_vectors(VECTORS, "en/friday"))
