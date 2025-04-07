from pathlib import Path

from tinlp.clf import PerceptronClassifier
from tinlp.utils.data import CorpusUNIMORPH, CorpusCONLL2003
from tinlp.utils.pfex import feat_morph_tag, feat_seq_conll

UNIMORPH_TRAIN = Path("./data/unimorph/swe-train")
UNIMORPH_TEST = Path("./data/unimorph/swe-test")
CONLL2003_TRAIN = Path("./data/conll2003/eng-20140.train")
CONLL2003_TEST = Path("./data/conll2003/eng-8201.testa")


def test_perceptron_morphological_clf():
    clf = PerceptronClassifier(
        feature_fn=feat_morph_tag,
        params={
            "metric": "accuracy",
            "epochs": 3,
            "affix_len": 5,
            "show_progress": True,
        },
    )
    X, y = CorpusUNIMORPH(
        UNIMORPH_TRAIN,
        delimiter="\t",
        fieldnames=["", "text", "label"],
    ).get_arrays(unsqueeze=True)
    clf.fit(X, y)

    X_test, y_test = CorpusUNIMORPH(
        UNIMORPH_TEST,
        delimiter="\t",
        fieldnames=["", "text", "label"],
    ).get_arrays(unsqueeze=True)
    print(f"{clf.metric.upper()}={clf.eval(X_test, y_test):.2f}")
    print(clf.get_top_k_features(10))


def test_perceptron_seq_clf():
    clf = PerceptronClassifier(
        feature_fn=feat_seq_conll,
        params={
            "metric": "accuracy",
            "epochs": 3,
            "eval_level": "token",
            "show_progress": True,
        },
    )
    clf.fit(*CorpusCONLL2003(CONLL2003_TRAIN).get_arrays())
    X_test, y_test = CorpusCONLL2003(CONLL2003_TEST).get_arrays()
    print(f"{clf.metric.upper()}={clf.eval(X_test, y_test):.2f}")
    print(clf.get_top_k_features(10))
