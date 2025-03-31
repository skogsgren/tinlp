from pathlib import Path

from tinlp.clf import PerceptronMorphClassifier
from tinlp.utils.data import get_data_split

TRAIN_DATA = Path("./data/unimorph/swedish-train-high")
TEST_DATA = Path("./data/unimorph/swedish-test")


def test_perceptron_morphological_clf():
    clf = PerceptronMorphClassifier(
        params={"metric": "accuracy", "epochs": 5, "affix_len": 5}
    )
    clf.fit(*get_data_split(TRAIN_DATA, data_type="unimorph"))
    score = clf.eval(*get_data_split(TEST_DATA, data_type="unimorph"))
    print(score)
    print(clf.weights.most_common(15))
    assert score > 0.5
