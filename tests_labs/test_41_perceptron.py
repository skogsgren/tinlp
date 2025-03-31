from pathlib import Path

from tinlp.clf import PerceptronMorphClassifier
from tinlp.utils.data import get_data_split

TRAIN_DATA = Path("./data/unimorph/swe-train")
TEST_DATA = Path("./data/unimorph/swe-test")


def test_perceptron_morphological_clf():
    print()
    clf = PerceptronMorphClassifier(
        params={"metric": "accuracy", "epochs": 3, "affix_len": 5}
    )
    clf.fit(*get_data_split(TRAIN_DATA, data_type="unimorph"))
    score = clf.eval(*get_data_split(TEST_DATA, data_type="unimorph"))
    for i in sorted(clf.weights.items(), reverse=True, key=lambda x: clf.weights[x[0]])[
        :10
    ]:
        print(i)
    # print(clf.weights.most_common(10))
    print(score)
    assert score >= 0.661
