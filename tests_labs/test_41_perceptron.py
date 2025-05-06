from pathlib import Path

from tinlp.clf import PerceptronClassifier
from tinlp.utils.pfex import feat_seq_conll, feat_morph_tag
from tinlp.utils.data import CorpusCONLL2003, CorpusUNIMORPH

UNIMORPH_TRAIN = Path("./data/unimorph/swe-train")
UNIMORPH_TEST = Path("./data/unimorph/swe-test")

CONLL2003_TRAIN = Path("./data/conll2003/eng.train")
CONLL2003_TEST = Path("./data/conll2003/eng.testa")
CONLL2003_TOKEN_ACC_LIMIT = 0.9


def test_perceptron_morphological_clf():
    clf = PerceptronClassifier(
        feature_fn=feat_morph_tag,
        params={
            "metric": "accuracy",
            "epochs": 3,
            "affix_len": 5,
            "show_progress": False,
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
    score = clf.eval(X_test, y_test)
    print(f"{clf.metric.upper()}={score:.2f}")
    print(clf.get_top_k_features(10))
    assert score >= 0.661


def test_perceptron_seq_clf():
    clf = PerceptronClassifier(
        feature_fn=feat_seq_conll,
        params={
            "metric": "accuracy",
            "eval_level": "token",
            "epochs": 3,
            "show_progress": False,
        },
    )
    X, y = CorpusCONLL2003(CONLL2003_TRAIN).get_arrays()
    clf.fit(X, y)

    X_test, y_test = CorpusCONLL2003(CONLL2003_TEST).get_arrays()
    score = clf.eval(X_test, y_test)
    print(f"{clf.metric.upper()}={score:.2f}")
    print(clf.get_top_k_features(10))
