from pathlib import Path

from tinynlp.clf import NaiveBayesClassifier

FILES = [x for x in Path("./data/sentiment_sentences/").iterdir() if x.suffix == ".tsv"]
LIMITS = {
    "imdb": 71.4 / 100,
    "amazon": 78.4 / 100,
    "yelp": 79.4 / 100,
}


def test_naive_bayes_clf():
    print()
    for file in FILES:
        clf = NaiveBayesClassifier(file)
        clf.fit()
        acc: float = clf.eval(file)
        print(f"{file}\t{acc:.2f} (LIMIT={LIMITS[file.stem]:.2f})")
        assert acc > LIMITS[file.stem]
