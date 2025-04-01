from pathlib import Path

from tinlp.clf import NaiveBayesClassifier
from tinlp.utils.data import CorpusCSV

DATA = Path("./data/cc_tsv/data.tsv")

TEST_SENTENCES = [
    (
        """Oh, you’ve gotta be kidding me. This? This is what we, as a society,
        are calling "pizza" now? I opened the box, and right away, I
        knew—something was off. The cheese looked like it had been applied with
        a malicious lack of care. The sauce distribution? Haphazard, like
        someone just gave up halfway through. And the crust? If I wanted to
        gnaw on something this dense, I’d take a bite out of my front door.
        And don’t even get me started on the baking instructions. 425 degrees?
        For what? To create a surface so unevenly cooked that half of it is
        burnt to a crisp and the other half is still thinking about it? And the
        smell—not the inviting aroma of a delicious pizza, no, but the stench
        of regret. The kind of smell that makes you stop and say, "Do I really
        need to be doing this with my life?" I took one bite and immediately
        felt betrayed. Betrayed by the box, by the brand, by whoever had the
        gall to call this a pizza. I wouldn’t serve this to my worst enemy. I
        wouldn’t serve this to someone I owe money to. You know what I did with
        the rest of it? I stared at it for a long time, sighed, and threw it
        directly into the garbage—where it belonged. Never again. Never.
        Again.""".replace("\n", " "),
        0,
    ),
]


def test_naive_bayes_clf():
    X, y = CorpusCSV(DATA, label_to_int=True, delimiter="\t").get_arrays()
    clf = NaiveBayesClassifier()
    clf.fit(X, y)
    for sentence, label in TEST_SENTENCES:
        pred = clf.predict(sentence)
        assert label == pred
