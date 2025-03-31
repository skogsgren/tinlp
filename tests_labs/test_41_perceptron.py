from pathlib import Path

from tinlp.clf import PerceptronClassifier
from tinlp.utils.data import CategorizedCorpus, get_data_split

UNIMORPH_TRAIN = Path("./data/unimorph/swe-train-5000")
UNIMORPH_TEST = Path("./data/unimorph/swe-test")

CONLL2003_TRAIN = Path("./data/conll2003/eng.train")
CONLL2003_TEST = Path("./data/conll2003/eng.testa")
CONLL2003_TOKEN_ACC_LIMIT = 0.9


def test_perceptron_morphological_clf():
    print()
    clf = PerceptronClassifier(
        params={
            "metric": "accuracy",
            "epochs": 3,
            "affix_len": 5,
            "feature_fn": "morph_tag",
            "show_progress": True,
        }
    )
    X, y = get_data_split(UNIMORPH_TRAIN, data_type="unimorph")
    X, y = [(n,) for n in X], [(n,) for n in y]
    clf.fit(X, y)

    X_test, y_test = get_data_split(UNIMORPH_TEST, data_type="unimorph")
    score = clf.eval(X_test, [(n,) for n in y_test])

    for w, c in clf.weights.most_common(10):
        print(w, c)
    print("score=", score)
    assert score >= 0.661


def test_perceptron_seq_clf():
    print()
    clf = PerceptronClassifier(
        params={
            "metric": "accuracy",
            "epochs": 3,
            "feature_fn": "seq_conll",
            "show_progress": True,
        }
    )
    cc = [x for x in CategorizedCorpus(CONLL2003_TRAIN, data_type="conll2003")]
    X = [i[0] for i in cc]
    y = [i[1] for i in cc]
    clf.fit(X, y)

    cc = [x for x in CategorizedCorpus(CONLL2003_TEST, data_type="conll2003")]
    X_test = [i[0] for i in cc]
    y_test = [i[1] for i in cc]
    score = clf.eval(X_test, y_test)  # seq accuracy

    total = 0
    correct = 0
    for i, seq_y_hat in enumerate(clf.predict(x) for x in X_test):
        for j, yhat in enumerate(seq_y_hat):
            total += 1
            if yhat == y_test[i][j]:
                correct += 1
    token_acc = correct / total
    assert CONLL2003_TOKEN_ACC_LIMIT <= token_acc

    for w, c in clf.weights.most_common(20):
        print(w, c)

    print(f"TOKEN_ACC={token_acc:.2f}")
    print(f"SEQ_ACC={score:.2f}")


print(test_perceptron_seq_clf())
