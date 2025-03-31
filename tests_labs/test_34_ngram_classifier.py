from pathlib import Path

from tinlp.clf import NGramClassifier
from tinlp.utils.data import get_data_split


SENT_CLASS_TRAIN = Path("./data/imdb/train/")
SENT_CLASS_TEST = Path("./data/imdb/test/")


def test_ngram_clf_lang_class():
    pass


def test_ngram_clf_sent_class():
    clf = NGramClassifier(params={"context": 1})
    clf.fit(*get_data_split(SENT_CLASS_TRAIN))
    print(clf.eval(*get_data_split(SENT_CLASS_TEST)))
