from pathlib import Path
import gzip
from typing import Iterable
import numpy as np


def get_ngrams(x: Iterable, pad: str | int, eos: str | int, n: int):
    """given an Iterable returns sliding window ngram iterable"""
    s = (pad,) * max((n - 1), 1) + tuple(x) + (eos,)
    for i in range(len(s) - n + 1):
        yield (s[i : i + n])


def read_vector_file(filename: Path) -> Iterable[tuple[str, np.ndarray]]:
    """helper function to read a vector file"""
    with (
        gzip.open(filename, "rt")
        if filename.suffixes[-1] == ".gz"
        else open(filename) as f
    ):
        for line in f:
            split_line = line.strip().split()
            if len(split_line) < 5:
                continue
            yield (split_line[0], np.array(split_line[1:], dtype=np.float32))


def search_vectors(vector_file: Path, query: str, k: int = 10) -> list:
    words = {}
    vectors = []
    for i, (word, vector) in enumerate(read_vector_file(vector_file)):
        words[word] = i
        vectors.append(vector)
    vectors = np.array(vectors)
    vectors_norm = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    if query not in words:
        return []
    query_vec = vectors[words[query]]
    query_norm = query_vec / np.linalg.norm(query_vec)

    cos_sin_distance = 1 - (vectors_norm @ query_norm)

    top_k = np.argsort(cos_sin_distance)[:k]
    inverted_vocab = {v: k for k, v in words.items()}
    top_words = [inverted_vocab[i] for i in top_k]
    print([(round(float(cos_sin_distance[i]), 2), inverted_vocab[i]) for i in top_k])
    return top_words
