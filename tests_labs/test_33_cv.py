from pathlib import Path

from tinlp.clf import NaiveBayesClassifier
from tinlp.eval import cross_val_score as cv
from tinlp.utils.data import CorpusCSV

TRAIN_DATA = Path("./data/sentiment_sentences/imdb.tsv")
ORIG = [68, 56, 84, 74, 76, 84, 74, 74, 70, 74]
AVG_SCORE = sum([x / 100 for x in ORIG]) / len(ORIG)


def test_cross_validation_naive_bayes():
    print()
    clf = NaiveBayesClassifier(params={"metric": "accuracy"})
    X, y = CorpusCSV(TRAIN_DATA, delimiter="\t").get_arrays()
    cv_score = cv(clf, X, y, k=10)
    avg_cv = sum(cv_score) / len(cv_score)
    assert avg_cv >= AVG_SCORE
