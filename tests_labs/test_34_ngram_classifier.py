from pathlib import Path

from tinlp.clf import NGramClassifier
from tinlp.utils.data import CorpusSubDir, CorpusCSV

CHAR_TRAIN_DATA = Path("./data/lang_identification/train.csv")
CHAR_TEST_DATA = Path("./data/lang_identification/test.csv")

SENT_CLASS_TRAIN = Path("./data/imdb/train/")
SENT_CLASS_TEST = Path("./data/imdb/test/")


def test_ngram_clf_lang_class():
    print()
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
    print("NGRAM_CHAR_CLF_ACC:", clf.eval(X_test, y_test))


def test_ngram_clf_sent_class():
    print()
    clf = NGramClassifier(params={"context": 1})
    clf.fit(*CorpusSubDir(SENT_CLASS_TRAIN).get_arrays())
    print("NGRAM_WORD_CLF_ACC=", clf.eval(*CorpusSubDir(SENT_CLASS_TEST).get_arrays()))
