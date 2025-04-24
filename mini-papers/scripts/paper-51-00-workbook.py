__doc__ = """contains all the code for reproducing paper (see README in data
folder for how to download data files). this should probably be a notebook
instead"""

import json
import tempfile
from collections import defaultdict
from pathlib import Path

from tinlp.clf import MultiNomialLRClassifier, PerceptronClassifier
from tinlp.eval import Metric, paired_bootstrap
from tinlp.utils.data import CorpusUNIMORPH
from tinlp.utils.pfex import feat_bpe, feat_morph_tag
from tinlp.utils.tokenizer import BPETokenizer

SEED = 42
DATA = [x for x in Path("./data/unimorph").iterdir() if x.is_file()]
BPE_MODEL_IN = Path(tempfile.mkstemp(suffix=".txt")[1])
BPE_MODEL_OUT = Path(tempfile.mkstemp(suffix=".json")[1])

BPE_SAMPLE_LINES = 1000
VOCAB_SIZE = 500  # intentionally low to only capture early merges
N_EPOCHS = 3
AFFIX_LEN = 5
BOOTSTRAP_SAMPLES = 5000

MODEL_METRICS_OUT = Path("./paper-51-comp-morph-model-choice-metrics.json")
FEATURE_METRICS_OUT = Path("./paper-51-comp-morph-feature-metrics.json")

print(f"BPE_MODEL_IN={BPE_MODEL_IN}")
print(f"BPE_MODEL_OUT={BPE_MODEL_OUT}")


# PART 1
def part1():
    metrics = {lang.name: defaultdict(dict) for lang in DATA}
    for lang_data in DATA:
        print(f"= {lang_data.name.upper()}")
        X_train, X_test, y_train, y_test = CorpusUNIMORPH(
            lang_data, delimiter="\t", fieldnames=["", "text", "label"]
        ).train_test_split(seed=42, unsqueeze=True)

        with open(BPE_MODEL_IN, "w") as f:
            for word, _ in (x[0] for x in X_train):
                f.write(f"{word}\n")
        bpe_tok = BPETokenizer(data=BPE_MODEL_IN, seed=SEED)
        bpe_tok.train(BPE_MODEL_OUT, vocab_size=VOCAB_SIZE, k=BPE_SAMPLE_LINES)
        print("finished training bpe")
        X_train_bpe = [((tuple(bpe_tok.tokenize(x[0][0])), x[0][1]),) for x in X_train]
        print("finished tokenizing trainset")
        X_test_bpe = [((tuple(bpe_tok.tokenize(x[0][0])), x[0][1]),) for x in X_test]
        print("finished tokenizing testset")

        for name, fn in [("bpe", feat_bpe), ("affix", feat_morph_tag)]:
            for statistic in ["accuracy", "f1"]:
                print(f"== {lang_data.name} {name} {statistic}".upper())
                params = {
                    "metric": statistic,
                    "epochs": N_EPOCHS,
                    "affix_len": AFFIX_LEN,
                    "show_progress": False,
                }
                clf_mn = MultiNomialLRClassifier(params, fn)
                clf_pn = PerceptronClassifier(params, fn)
                if name == "bpe":
                    iter_X_train = X_train_bpe
                    iter_X_test = X_test_bpe
                else:
                    iter_X_train = X_train
                    iter_X_test = X_test

                clf_mn.fit(iter_X_train, y_train)
                clf_pn.fit(iter_X_train, y_train)

                mn_pred = [clf_mn.predict(x) for x in iter_X_test]
                pn_pred = [clf_pn.predict(x) for x in iter_X_test]

                metric = Metric(statistic)

                metrics[lang_data.name][f"{name}_{statistic}"]["perceptron"] = metric(
                    y_test, pn_pred
                )
                metrics[lang_data.name][f"{name}_{statistic}"]["multinomial"] = metric(
                    y_test, mn_pred
                )
                print("finished training/evaluating classifiers")

                print(
                    f"performing paired bootstrap between multinomial/perceptron for {statistic}"
                )
                print(
                    "i.e. H0 = there is no difference between the performance of a multinomial compared to perceptron model"
                )
                p_value, ci = paired_bootstrap(
                    clf_one_pred=mn_pred,
                    clf_two_pred=pn_pred,
                    statistic=params["metric"],
                    test_y=y_test,
                    n=BOOTSTRAP_SAMPLES,
                )
                metrics[lang_data.name][f"{name}_bootstrap"][f"mn_pn_{statistic}"] = (
                    p_value,
                    ci,
                )

        with open(MODEL_METRICS_OUT, "w") as f:
            json.dump(metrics, f)


# PART 2
def part2():
    metrics = {lang.name: defaultdict(dict) for lang in DATA}
    print("= AFFIX/BPE PERFORMANCE")
    for lang_data in DATA:
        print(f"= {lang_data.name.upper()}")
        X_train, X_test, y_train, y_test = CorpusUNIMORPH(
            lang_data, delimiter="\t", fieldnames=["", "text", "label"]
        ).train_test_split(seed=42, unsqueeze=True)

        with open(BPE_MODEL_IN, "w") as f:
            for word, _ in (x[0] for x in X_train):
                f.write(f"{word}\n")
        bpe_tok = BPETokenizer(data=BPE_MODEL_IN, seed=SEED)
        bpe_tok.train(BPE_MODEL_OUT, vocab_size=VOCAB_SIZE, k=BPE_SAMPLE_LINES)
        print("finished training bpe")
        X_train_bpe = [((tuple(bpe_tok.tokenize(x[0][0])), x[0][1]),) for x in X_train]
        print("finished tokenizing trainset")
        X_test_bpe = [((tuple(bpe_tok.tokenize(x[0][0])), x[0][1]),) for x in X_test]
        print("finished tokenizing testset")
        # since perceptron is much quicker, and none of the bootstrap tests have
        # high effect sizes, we just use the perceptron to do paired bootstrap
        # between bpe and affix for feature choice
        params = {
            "epochs": N_EPOCHS,
            "affix_len": AFFIX_LEN,
            "show_progress": False,
        }
        clf_bpe = PerceptronClassifier(params, feat_bpe)
        clf_bpe.fit(X_train_bpe, y_train)
        clf_affix = PerceptronClassifier(params, feat_morph_tag)
        clf_affix.fit(X_train, y_train)
        print("finished training models for bpe/affix comparison")
        for statistic in ["accuracy", "f1"]:
            metric = Metric(statistic)

            affix_pred = [clf_affix.predict(x) for x in X_test]
            bpe_pred = [clf_bpe.predict(x) for x in X_test_bpe]

            affix_score = metric(y_test, affix_pred)
            bpe_score = metric(y_test, bpe_pred)

            metrics[lang_data.name]["bpe"][statistic] = bpe_score
            metrics[lang_data.name]["affix"][statistic] = affix_score

            print(f"calculating bootstrap between affix and bpe for {statistic}")
            print(
                """i.e. H0 = there is no significant difference between a model
                trained on affix-based features compared to low-vocab BPE"""
            )

            p_value, ci = paired_bootstrap(
                clf_one_pred=affix_pred,
                clf_two_pred=bpe_pred,
                statistic=statistic,
                test_y=y_test,
                n=BOOTSTRAP_SAMPLES,
            )
            metrics[lang_data.name]["affix_bpe_bootstrap"][statistic] = (
                p_value,
                ci,
            )

            print(f"finished bootstrap between bpe and affix for {statistic}")

    with open(FEATURE_METRICS_OUT, "w") as f:
        json.dump(metrics, f)


part1()
part2()
