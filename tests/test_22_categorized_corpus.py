from pathlib import Path
from csv import DictReader
import json

from tinynlp.utils.data import CategorizedCorpus

SUBDIR_DATA = Path("data/cc_subdir")
TSV_DATA = Path("data/cc_tsv/data.tsv")


def test_subdir_categorized_corpus():
    with open(SUBDIR_DATA / "EXPECTED.json") as f:
        EXPECTED = [(str(x[0]), int(x[1])) for x in json.load(f)]
    cc = CategorizedCorpus(SUBDIR_DATA)
    for y, y_hat in zip(EXPECTED, cc):
        assert y == y_hat


def test_tsv_categorized_corpus():
    with open(TSV_DATA, newline="") as f:
        EXPECTED = [
            (str(line["text"]), int(line["label"]))
            for line in DictReader(f, delimiter="\t", fieldnames=["text", "label"])
        ]
    y_hat = [x for x in CategorizedCorpus(TSV_DATA)]
    assert EXPECTED == y_hat
