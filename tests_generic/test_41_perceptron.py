from pathlib import Path

from tinlp.clf import PerceptronClassifier
from tinlp.utils.data import CorpusCSV, CorpusCONLL2003
from tinlp.utils.pfex import feat_morph_tag, feat_seq_conll

UNIMORPH_TRAIN = Path("./data/unimorph/swe-train")
UNIMORPH_TEST = Path("./data/unimorph/swe-test")
CONLL2003_TRAIN = Path("./data/conll2003/eng-20140.train")
CONLL2003_TEST = Path("./data/conll2003/eng-8201.testa")
CONLL2003_TOKEN_ACC_LIMIT = 0.90


def test_perceptron_morphological_clf():
    print()
    clf = PerceptronClassifier(
        feature_fn=feat_morph_tag,
        params={
            "metric": "accuracy",
            "epochs": 3,
            "affix_len": 5,
            "show_progress": True,
        },
    )
    X, y = CorpusCSV(
        UNIMORPH_TRAIN,
        delimiter="\t",
        fieldnames=["", "text", "label"],
    ).get_arrays(unsqueeze=True)
    clf.fit(X, y)

    X_test, y_test = CorpusCSV(
        UNIMORPH_TEST,
        delimiter="\t",
        fieldnames=["", "text", "label"],
    ).get_arrays(unsqueeze=True)

    score = clf.eval(X_test, y_test)

    for w, c in clf.weights.most_common(10):
        print(w, c)
    print("score=", score)
    assert score >= 0.661


def test_perceptron_seq_clf():
    print()
    clf = PerceptronClassifier(
        feature_fn=feat_seq_conll,
        params={
            "metric": "accuracy",
            "epochs": 3,
            "show_progress": True,
        },
    )
    X, y = CorpusCONLL2003(CONLL2003_TRAIN).get_arrays()
    clf.fit(X, y)

    X_test, y_test = CorpusCONLL2003(CONLL2003_TEST).get_arrays()
    score = clf.eval(X_test, y_test)  # seq accuracy

    total = 0
    correct = 0
    for i, seq_y_hat in enumerate(clf.predict(x) for x in X_test):
        for j, yhat in enumerate(seq_y_hat):
            total += 1
            if yhat == y_test[i][j]:
                correct += 1
    token_acc = correct / total

    for w, c in clf.weights.most_common(20):
        print(w, c)

    print(f"TOKEN_ACC={token_acc:.2f}")
    print(f"SEQ_ACC={score:.2f}")
    assert CONLL2003_TOKEN_ACC_LIMIT <= token_acc
