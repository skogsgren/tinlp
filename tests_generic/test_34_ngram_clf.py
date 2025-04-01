from pathlib import Path

from tinlp.clf import NGramClassifier
from tinlp.utils.data import CorpusCSV

WORD_TRAIN_DATA = Path("./data/cc_tsv/data.tsv")
WORD_TEST_DATA = Path("./data/cc_tsv/test.tsv")
WORD_ACC = 0.8

CHAR_TRAIN_DATA = Path("./data/lang_identification/train.csv")
CHAR_TEST_DATA = Path("./data/lang_identification/test.csv")
CHAR_ACC = 0.99


def test_ngram_clf_word():
    clf = NGramClassifier(
        {
            "ngram_size": 2,
            "ngram_smoothing": 0.001,
            "tokenizer": "regex",
            "metric": "accuracy",
        }
    )
    X, y = CorpusCSV(WORD_TRAIN_DATA, delimiter="\t").get_arrays()
    clf.fit(X, y)
    X_test, y_test = CorpusCSV(WORD_TEST_DATA, delimiter="\t").get_arrays()
    word_test_acc = clf.eval(X_test, y_test)
    print("NGRAM WORD CLF ACC:", word_test_acc)
    assert word_test_acc == WORD_ACC


def test_ngram_clf_char():
    clf = NGramClassifier(
        {
            "ngram_size": 1,
            "ngram_smoothing": 0.001,
            "tokenizer": "char",
            "metric": "accuracy",
        }
    )
    X, y = CorpusCSV(CHAR_TRAIN_DATA).get_arrays()
    clf.fit(X, y)
    X_test, y_test = CorpusCSV(CHAR_TEST_DATA).get_arrays()
    char_test_acc = clf.eval(X_test, y_test)
    print("NGRAM CHARACTER CLF ACC:", char_test_acc)
    assert char_test_acc == CHAR_ACC
