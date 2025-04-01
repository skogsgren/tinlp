from tinlp.clf import NaiveBayesClassifier
from tinlp.utils.data import CorpusCSV

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
        clf.fit(*CorpusCSV(train, delimiter="\t", label_to_int=True).get_arrays())
        acc = clf.eval(*CorpusCSV(test, delimiter="\t", label_to_int=True).get_arrays())
        print(f"{domain}: {acc:.2f} (LIMIT={LIMITS[domain]:.2f})")
        assert acc >= LIMITS[domain]
