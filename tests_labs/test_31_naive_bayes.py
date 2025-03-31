from tinlp.clf import NaiveBayesClassifier
from tinlp.utils.data import get_data_split

LIMITS = {
    "imdb": 71.4 / 100,
    "amazon": 78.4 / 100,
    "yelp": 79.4 / 100,
}
FILES = {}
for k in LIMITS:
    FILES[k] = (
        f"./data/sentiment_sentences/{k}-train.tsv",
        f"./data/sentiment_sentences/{k}-test.tsv",
    )


def test_naive_bayes_clf():
    print()
    for domain, (train, test) in FILES.items():
        clf = NaiveBayesClassifier({"metric": "accuracy"})
        clf.fit(*get_data_split(train))
        acc: float = clf.eval(*get_data_split(test))
        print(f"{acc:.2f} (LIMIT={LIMITS[domain]:.2f})")
        assert acc >= LIMITS[domain]
