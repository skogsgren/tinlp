__doc__ = """code to parse the metrics json in order to create tables in LaTeX paper using templates"""

import json
import sys
from pathlib import Path

DIR = Path(__file__).parent
MODEL_CHOICE_METRICS_FN = DIR / "./paper-51-comp-morph-model-choice-metrics.json"
FEATURE_METRICS_FN = DIR / "./paper-51-comp-morph-feature-metrics.json"
LANG_MAPPINGS = {"cym": "Welsh", "pei": "Chichimeca", "eus": "Basque"}

with open(FEATURE_METRICS_FN, "r") as f:
    feature_metrics = json.load(f)
with open(MODEL_CHOICE_METRICS_FN, "r") as f:
    model_choice_metics = json.load(f)


def show_feature_metrics():
    for metric in ["accuracy", "f1"]:
        MODEL_METRICS_TABLE = f"""\\begin{{table*}}[ht]
\\centering
\\begin{{tabular}}{{lrrrrc}}
\\textbf{{Language}} & Affix {metric.capitalize()} & BPE {metric.capitalize()} & $\\delta$ & $p$ & 95\\% CI\\\\
\\hline
"""
        for lang_code, results in feature_metrics.items():
            lang = f"\\texttt{{{LANG_MAPPINGS[lang_code]}}}"
            aff_m = results["affix"][metric]
            bpe_m = results["bpe"][metric]
            delta = float(aff_m) - float(bpe_m)
            p = round(results["affix_bpe_bootstrap"][metric][0], 3)
            if p == 0.0:
                p = "<0.001"
            ci = tuple(round(x, 3) for x in results["affix_bpe_bootstrap"][metric][1])
            ci = f"$\\begin{{bmatrix}}{ci[0]} & {ci[1]}\\end{{bmatrix}}$"

            MODEL_METRICS_TABLE += (
                f"{lang} & {aff_m:.2f} & {bpe_m:.2f} & {delta:.2f} & {p} & {ci} \\\\\n"
            )
        MODEL_METRICS_TABLE += f"""\\end{{tabular}}
\\caption{{{metric.capitalize()} tagging performance with paired bootstrap test ($H_0$: no difference between affix and BPE-based features)}}
\\label{{tab:feat-{metric}}}
\\end{{table*}}
"""
        print(MODEL_METRICS_TABLE)


def show_model_choice_metrics():
    MODEL_CHOICE_TABLE = ""
    for feature_fn in ["affix", "bpe"]:
        print()
        for metric in ["accuracy", "f1"]:
            MODEL_CHOICE_TABLE = """\\begin{table}[ht]
\\centering
\\begin{tabular}{lrrrc}
\\textbf{Language} & $\\delta$ & $p$ & \\textbf{95\\% CI}\\\\
\\hline
"""
            for lang, results in model_choice_metics.items():
                lang = LANG_MAPPINGS[lang]
                mn_m = results[f"{feature_fn}_{metric}"]["multinomial"]
                pn_m = results[f"{feature_fn}_{metric}"]["perceptron"]
                delta = round(mn_m - pn_m, 5)
                p = round(results[f"{feature_fn}_bootstrap"][f"mn_pn_{metric}"][0], 3)
                if p == 0.0:
                    p = "<0.001"
                ci = tuple(
                    round(x, 3)
                    for x in results[f"{feature_fn}_bootstrap"][f"mn_pn_{metric}"][1]
                )
                ci = f"$\\begin{{bmatrix}}{ci[0]} & {ci[1]}\\end{{bmatrix}}$"
                MODEL_CHOICE_TABLE += f"{lang} & {delta} & {p} & {ci} \\\\\n"
            fmt_feat_fn = "BPE" if feature_fn == "BPE" else "affix"
            MODEL_CHOICE_TABLE += f"""\\end{{tabular}}
\\caption{{1-tailed bootstrap test. $H_0$: using {fmt_feat_fn} features and {metric} for metrics, is the multinomial logistic regression model significantly better than the perceptron model?}}
\\label{{tab:{feature_fn}-{metric}-mc}}
\\end{{table}}"""
            print(MODEL_CHOICE_TABLE)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        show_feature_metrics()
        show_model_choice_metrics()
    elif sys.argv[1] == "-f":
        show_feature_metrics()
    elif sys.argv[1] == "-m":
        show_model_choice_metrics()
