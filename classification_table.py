import re
import json
import sys
from pathlib import Path

import pandas as pd

# python classification_table.py [results_dir] [table_name]
#   default: outputs/pair_classifier -> reverse_table.tex
#   temperature: python classification_table.py outputs/pair_classifier_temperature reverse_table_temperature.tex
p = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/raid/homes/giovanni.puccetti/Repos/recreate/outputs/pair_classifier/")
table_name = sys.argv[2] if len(sys.argv) > 2 else "reverse_table.tex"
experiments = [i for i in p.iterdir() if i.is_dir()]
experiments_models = []
for i in experiments:
    experiments_models += [j for j in i.iterdir() if "reverse" in str(j)]
future_df = pd.DataFrame([])
for exp_mod in experiments_models:
    results_file = exp_mod / "results.json"
    if not results_file.exists():
        continue
    with open(results_file) as f:
        exp_mod_results = json.load(f)["test"]
    # number of correct test predictions (overall and per gap), so models can be pooled exactly:
    # per-gap test sizes come from the saved test pairs, accuracy * size gives back the integer count
    test_pairs = pd.read_csv(exp_mod / "pairs.csv").query("split == 'test'")
    n_gap = test_pairs["gap"].value_counts()
    n_test = sum(map(sum, exp_mod_results.pop("confusion")))
    assert n_test == len(test_pairs), exp_mod
    for key in [k for k in exp_mod_results if k == "accuracy" or k.startswith("acc_gap")]:
        n = n_test if key == "accuracy" else n_gap[int(key.removeprefix("acc_gap"))]
        correct = exp_mod_results[key] * n
        assert abs(correct - round(correct)) < 1e-6, (exp_mod, key)
        exp_mod_results[f"correct_{key}"], exp_mod_results[f"n_{key}"] = round(correct), n
    exp_mod_results_df = pd.DataFrame.from_dict([exp_mod_results])
    # odd-step runs (<model>_reverse_alternate) become a separate experiment, e.g. acl -> acl_alternate
    alternate = exp_mod.name.endswith("_alternate")
    exp_mod_results_df["experiment"] = exp_mod.parent.name + ("_alternate" if alternate else "")
    exp_mod_results_df["model"] = exp_mod.name.removesuffix("_alternate").removesuffix("_reverse")
    future_df = pd.concat([future_df, exp_mod_results_df], axis=0)
future_df = future_df.set_index(['experiment', 'model',]).sort_index()

# overall accuracy, then per-gap accuracies in numerical order (-10, ..., -2, 2, ..., 10)
gap_columns = sorted([i for i in future_df.columns if i.startswith("acc_gap")], key=lambda c: int(c.removeprefix("acc_gap")))
# pool the models' test sets: per experiment, total correct predictions / total test pairs (overall and per gap)
acc_columns = ["accuracy"] + gap_columns
totals = future_df.groupby(level="experiment").sum(numeric_only=True)
table = pd.DataFrame({c: totals[f"correct_{c}"] / totals[f"n_{c}"] for c in acc_columns}) * 100
print(future_df.reset_index().groupby("experiment")["model"].agg(list).to_string())   # models pooled per experiment
# per-generator table: each run's own test accuracies, one row per (experiment, generator)
per_generator = future_df[acc_columns] * 100
column_names = ["Acc."] + [f"${c.removeprefix('acc_gap')}$" for c in gap_columns]
table.columns = column_names
per_generator.columns = column_names
dataset_names = {"acl": "ACL", "bbc-news": "BBC News", "erc-grants": "ERC Grants"}


def experiment_name(e):
    base = e.removesuffix("_alternate")
    name = dataset_names.get(base, base)
    if e.endswith("_alternate"):
        # line break inside the \multirow cell (needs \usepackage{makecell})
        name = f"\\makecell[l]{{{name}\\\\(odd steps)}}"
    return name


def to_latex(df, caption, label, column_format):
    # LaTeX table (needs \usepackage{booktabs,makecell,multirow,adjustbox}), values in %
    latex = (
        df.style
        .format("{:.1f}", na_rep="--")
        .to_latex(
            hrules=True,
            column_format=column_format,
            position="t",
            position_float="centering",
            caption=caption,
            label=label,
            multirow_align="c",
            clines="skip-last;data",
        )
    )
    latex = latex.replace("{table}", "{table*}")
    latex = re.sub(r"\\cline\{[^}]*\}\n(?=\\bottomrule)", "", latex)   # no rule right above \bottomrule
    latex = latex.replace("\\begin{tabular}", "\\begin{adjustbox}{max width=\\linewidth}\n\\begin{tabular}")
    latex = latex.replace(r"\end{tabular}", "\\end{tabular}\n\\end{adjustbox}")
    return latex


caption = ("Pair classifier test accuracy (\\%) with reversed pairs, overall and per gap "
           "(negative gaps: the two texts are swapped)")
# default table keeps its label, others get one from the file name (e.g. tab:reverse_table_temperature)
label = "tab:pair_classifier_reverse" if table_name == "reverse_table.tex" else f"tab:{Path(table_name).stem}"
w = Path("/raid/homes/giovanni.puccetti/Latex/acl_content_convergence/tables/")

# 1) aggregated: one row per experiment, pooled over the generators' test sets
table = table.rename(index=experiment_name)
table.index.name = None
print("\n=== Aggregated (pooled over generators) ===")
print(table.round(1).to_string())
latex = to_latex(table, caption + ", pooled over the test sets of all generator models.", label,
                 "l" + "c" * len(table.columns))
print(latex)
(w / table_name).write_text(latex)
print(f"wrote {w / table_name}")

# 2) per generator: one row per (experiment, generator)
per_generator = per_generator.rename(index=experiment_name, level="experiment")
per_generator.index.names = [None, None]
per_generator_name = f"{Path(table_name).stem}_per_generator.tex"
print("\n=== Per generator ===")
print(per_generator.round(1).to_string())
latex = to_latex(per_generator, caption + ", for each generator model separately.", f"{label}_per_generator",
                 "ll" + "c" * len(per_generator.columns))
print(latex)
(w / per_generator_name).write_text(latex)
print(f"wrote {w / per_generator_name}")
