from pathlib import Path

from tinlp.utils.text import search_vectors


VECTORS = Path("./data/glove-en.vec.gz")


def test_vector_similarity_fn():
    print(search_vectors(VECTORS, "en/friday"))


test_vector_similarity_fn()
