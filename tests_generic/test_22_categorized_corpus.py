import json
from csv import DictReader
from pathlib import Path

from tinlp.utils.data import (
    CorpusCONLL2003,
    CorpusCSV,
    CorpusPlain,
    CorpusSubDir,
)

SUBDIR_DATA = Path("data/cc_subdir")
TSV_DATA = Path("data/cc_tsv/data.tsv")
CSV_DATA = Path("./data/lang_identification/test.csv")
CONLL_DATA = Path("./data/conll2003/eng-8201.testa")
PLAIN_DATA = Path("./data/bpe_train.txt")
UNIMORPH_DATA = Path("./data/unimorph/swe-test")


def test_subdir_categorized_corpus():
    with open(SUBDIR_DATA / "EXPECTED.json") as f:
        EXPECTED = [(str(x[0]), int(x[1])) for x in json.load(f)]
        y = [x[1] for x in EXPECTED]
    assert CorpusSubDir(SUBDIR_DATA).get_arrays()[1] == y
    for y, y_hat in zip(EXPECTED, CorpusSubDir(SUBDIR_DATA)):
        assert y == y_hat


test_subdir_categorized_corpus()


def test_tsv_categorized_corpus():
    with open(TSV_DATA, newline="") as f:
        EXPECTED = [
            (str(line["text"]), int(line["label"]))
            for line in DictReader(f, delimiter="\t", fieldnames=["text", "label"])
        ]
        y = [x[1] for x in EXPECTED]
    y_hat = [x for x in CorpusCSV(TSV_DATA, delimiter="\t", label_to_int=True)]
    assert EXPECTED == y_hat
    assert CorpusCSV(TSV_DATA, delimiter="\t", label_to_int=True).get_arrays()[1] == y


def test_csv_categorized_corpus():
    with open(CSV_DATA, newline="") as f:
        EXPECTED = [
            (str(line["text"]), line["label"])
            for line in DictReader(f, fieldnames=["text", "label"])
        ]
        y = [x[1] for x in EXPECTED]
    y_hat = [x for x in CorpusCSV(CSV_DATA)]
    assert EXPECTED == y_hat
    assert CorpusCSV(CSV_DATA).get_arrays()[1] == y


def test_tsv_categorized_corpus_column_specification():
    with open(UNIMORPH_DATA, newline="") as f:
        EXPECTED = [
            (str(line["text"]), line["label"])
            for line in DictReader(
                f, delimiter="\t", fieldnames=["lemma", "text", "label"]
            )
        ]
    y_hat = [
        x
        for x in CorpusCSV(
            UNIMORPH_DATA,
            delimiter="\t",
            fieldnames=["lemma", "text", "label"],
        )
    ]
    assert EXPECTED == y_hat


def test_conll2003():
    EXPECTED = []
    with open(CONLL_DATA.with_suffix(".json"), "r") as f:
        raw = json.load(f)
    for x, y in (tuple(x) for x in raw):
        EXPECTED.append((tuple(tuple(i) for i in x), tuple(y)))
    EXPECTED = tuple(EXPECTED)
    for i, yhat in enumerate(CorpusCONLL2003(CONLL_DATA)):
        assert EXPECTED[i] == yhat


def test_plain_corpus():
    with open(PLAIN_DATA) as f:
        EXPECTED = [x.strip() for x in f]
    for i, row in enumerate(CorpusPlain(PLAIN_DATA)):
        assert EXPECTED[i] == row
