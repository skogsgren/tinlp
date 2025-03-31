from pathlib import Path

from tinlp.clf import NaiveBayesClassifier
from tinlp.utils.data import get_data_split
from tinlp.eval import cross_val_score as cv

DATA = Path("./data/cc_tsv/data.tsv")
SCORES = [0.9333, 0.8667]


def test_cross_validation():
    clf = NaiveBayesClassifier()
    X, y = get_data_split(DATA)
    cv_score = cv(clf, X, y, k=2)
    for i, score in enumerate(cv_score):
        assert round(score, 4) == SCORES[i]
