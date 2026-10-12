"""Collect every results/summary_*.json into one Markdown table (results/summary_table.md).

Usage:  python summarize.py
"""
import json
from config import RESULTS_DIR

RUNS = [  # (summary tag, description)
    ("inv_mse", "paper: inverse-MSE likelihood"),
    ("gauss_refine", "Gaussian likelihood + LM refine"),
    ("gauss_softprop", "Gaussian likelihood + softmax proposal"),
    ("gauss_refine_los", "Gaussian likelihood + LM refine, known LOS"),
    ("gauss_sig1_softprop_los", "Gaussian (sigma 1 px) + softmax proposal, known LOS"),
]

lines = ["| Run | Method | Position RMSE [m] | Attitude error [deg] |", "|---|---|---|---|"]
for tag, desc in RUNS:
    path = RESULTS_DIR / f"summary_{tag}.json"
    if not path.exists():
        print(f"missing {path.name} - skipped")
        continue
    for name, m in json.load(open(path)).items():
        lines.append(f"| {desc} | {name} | {m['pos_rmse']:.2f} ± {m['pos_rmse_std']:.2f} "
                     f"| {m['att_mean_deg']:.2f} ± {m['att_mean_deg_std']:.2f} |")
(RESULTS_DIR / "summary_table.md").write_text(
    "Mean ± std over 10 Monte-Carlo runs of the 10 s approach scenario.\n\n" + "\n".join(lines) + "\n",
    encoding="utf-8")
print("\n".join(lines))
print(f"\nsaved {RESULTS_DIR / 'summary_table.md'}")
