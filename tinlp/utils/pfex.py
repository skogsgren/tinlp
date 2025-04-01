__doc__ = """contains examples of get_features for perceptron classifier"""

from typing import Iterable


def feat_seq_conll(i: int, seq_x: str, seq_y: str) -> Iterable[str]:
    """function for generating perceptron features for NER using CONLL2003 data"""
    # context features
    if i != 0:
        yield f"{seq_y[i]} prev={seq_x[i - 1]}"
    yield f"{seq_y[i]} curr={seq_x[i]}"
    if (i + 1) != len(seq_y):
        yield f"{seq_y[i]} next={seq_x[i + 1]}"
    # word features
    yield f"{seq_y[i]} has_capital={any(c.isupper() for c in seq_x[i][0][0])}"


def feat_morph_tag(i: int, seq_x: tuple, seq_y: tuple, k: int = 5) -> Iterable[str]:
    """generates morphological UNIMORPH tags using affixes. assumes 2D data in 3D
    format (e.g. ['word'] not just 'word')"""
    if len(seq_x) != 1 or len(seq_y) != 1:
        raise ValueError("Data length != 1 (Make sure data is 3D)")
    pos, tag = seq_y[0].split(";")[0], ";".join(seq_y[0].split(";")[1:])
    yield f"{pos} has_capital={any(c.isupper() for c in seq_x[0])} tag={tag}"
    for i in range(1, min(k + 1, len(seq_x[0]) + 1)):
        yield f"{pos} prefix={seq_x[0][:i]} tag={tag}"
        yield f"{pos} suffix={seq_x[0][-i:]} tag={tag}"
