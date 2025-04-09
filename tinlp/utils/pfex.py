__doc__ = """contains examples of get_features for perceptron classifier"""

from typing import Iterable


def feat_seq_conll(self, seq_x: str, i: int) -> Iterable[str]:
    """function for generating perceptron features for NER using CONLL2003 data"""
    # context features
    if i != 0:
        yield f"prev={seq_x[i - 1]}"
    yield f"curr={seq_x[i]}"
    if (i + 1) != len(seq_x):
        yield f"next={seq_x[i + 1]}"

    # word features
    yield f"has_capital={any(c.isupper() for c in seq_x[i][0][0])}"


def feat_morph_tag(self, seq_x: tuple, i: int) -> Iterable[str]:
    for i in range(1, min(self.params.get("affix_len", 5) + 1, len(seq_x[0][0]) + 1)):
        yield f"prefix={seq_x[0][0][:i]}"
        yield f"suffix={seq_x[0][0][-i:]}"
    yield f"pos={seq_x[0][1]}"


def feat_bpe(self, seq_x: tuple, i: int) -> Iterable[str]:
    for subword in seq_x[0][0]:
        yield subword
    yield seq_x[0][1]
