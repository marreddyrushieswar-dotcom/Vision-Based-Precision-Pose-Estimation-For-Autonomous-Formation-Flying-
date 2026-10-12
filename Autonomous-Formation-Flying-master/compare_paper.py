"""Plot the paper's published results (digitised from its Figs. 9 / 10) against ours.

Usage:  python compare_paper.py
Needs:  paper_data/paper_fig*.csv (python digitize_paper.py) and the results/ files from
        python simulate.py
        python simulate.py --likelihood gauss --refine
        python simulate.py --likelihood gauss --refine --known-los
        python simulate.py --likelihood gauss --proposal softmax
        python simulate.py --likelihood gauss --proposal softmax --sigma-px 1 --known-los
Output: results/paper_vs_ours_fig9.png, results/paper_vs_ours_fig10.png,
        results/paper_vs_ours_summary.png
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import *

PAPER_DIR = ROOT / "paper_data"
PAPER_TOP1 = 0.54                 # paper Section IV-D
OURS_TOP1_NO_ATT = 0.643          # side experiment: same 100-100 ANN, leader pitch/yaw fixed at 0

# fixed series order -> fixed colours (validated categorical palette, light mode)
SERIES = [  # (label, short label, colour, results tag or None for the paper)
    ("Paper (Punnoose)", "Paper", "#2a78d6", None),
    ("Ours - paper method", "Ours, paper method", "#eb6834", "inv_mse"),
    ("Ours - improved PF", "Ours, improved", "#1baf7a", "gauss_softprop"),
    ("Ours - improved PF + known LOS", "Ours, improved + LOS", "#eda100", "gauss_sig1_softprop_los"),
]
CONFIGS = {"fig9": ("PF N=1000, a=0.9 every 10 cycles", "N = 1000, α = 0.9 every 10 cycles"),
           "fig10": ("PF N=5000, a=0.9 every cycle", "N = 5000, α = 0.9 every cycle")}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": INK,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
                     "axes.axisbelow": True, "legend.frameon": False})


def load_paper(name):
    d = np.loadtxt(PAPER_DIR / f"paper_{name}.csv", delimiter=",", skiprows=1)
    return d[:, 0], d[:, 1], d[:, 2]


def load_ours(tag, method):
    """First Monte-Carlo run of `method` from results/estimates_<tag>.npz -> t, pos err, att err."""
    from simulate import errors
    d = np.load(RESULTS_DIR / f"estimates_{tag}.npz")
    pe, ae = errors(d[method.replace(" ", "_")], d["truth"])
    return d["t"], pe, ae


def summary(tag):
    return json.load(open(RESULTS_DIR / f"summary_{tag}.json"))


def end_labels(ax, ends, log=False):
    """Direct labels (series name + mean) at the right end of each line, nudged apart so they
    don't collide; a colour chip beside the text carries the identity, the text stays neutral."""
    f = (lambda v: np.log10(max(v, 1e-3))) if log else (lambda v: v)
    finv = (lambda v: 10 ** v) if log else (lambda v: v)
    lo, hi = (f(v) for v in ax.get_ylim()); gap = 0.075 * (hi - lo)
    placed = []
    for x, y, txt, colour in sorted(ends, key=lambda e: f(e[1])):
        y_lab = max(f(y), placed[-1] + gap) if placed else f(y)
        placed.append(y_lab)
        ax.plot([x + 0.12], [finv(y_lab)], "s", ms=6, color=colour, clip_on=False)
        ax.annotate(txt, (x + 0.25, finv(y_lab)), fontsize=8, color=INK, va="center",
                    annotation_clip=False)


def time_series(name):
    method, title = CONFIGS[name]
    fig, ax = plt.subplots(2, 1, figsize=(9.5, 6.2), sharex=True)
    ends = ([], [])
    for label, short, colour, tag in SERIES:
        t, pe, ae = load_paper(name) if tag is None else load_ours(tag, method)
        for k, (e, unit) in enumerate(((pe, "m"), (ae, "°"))):
            ax[k].plot(t, e, color=colour, lw=1.6, label=label)
            ends[k].append((t[-1], e[-1], f"{short}: mean {np.nanmean(e):.2f} {unit}", colour))
    ax[0].set_yscale("log"); ax[0].set_ylim(0.01, 50)
    ax[0].set_yticks([0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30], ["0.01", "0.03", "0.1", "0.3", "1", "3", "10", "30"])
    ax[1].set_ylim(bottom=0)
    for k in range(2):
        end_labels(ax[k], ends[k], log=(k == 0))
    ax[0].set_ylabel("position error [m], log scale"); ax[1].set_ylabel("attitude error [deg]")
    ax[1].set_xlabel("time [s]"); ax[1].set_xlim(0, 10.1)
    ax[0].legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=4, fontsize=8, handlelength=1.5)
    fig.suptitle(f"Paper {name.replace('fig', 'Fig. ')} vs ours: PF {title}", color=INK, y=0.995)
    fig.text(0.01, 0.005, "Paper: digitised from the published figure (one run). Ours: first of 10 "
             "Monte-Carlo runs; the means are for the plotted run.", fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 0.76, 1)); fig.savefig(RESULTS_DIR / f"paper_vs_ours_{name}.png", dpi=150)
    plt.close(fig)


def bar_labels(ax, bars, fmt, err=None):
    """Value above each bar (above its error bar, if any)."""
    err = np.zeros(len(bars)) if err is None else err
    for b, e in zip(bars, err):
        h = b.get_height()
        if np.isfinite(h):
            ax.annotate(fmt.format(h), (b.get_x() + b.get_width() / 2, h + e), xytext=(0, 2),
                        textcoords="offset points", ha="center", va="bottom", fontsize=7, color=INK)


def ours_top1():
    try:
        import ann
        va = np.load(DATA_DIR / "val.npz")
        return float((ann.predict(ann.load(), va["X"]) == va["y"]).mean())
    except FileNotFoundError:          # data/ is not committed; fall back to the reported value
        return 0.3358


def summary_figure():
    fig, ax = plt.subplots(1, 3, figsize=(15, 5), gridspec_kw={"width_ratios": [1, 1.7, 1.7]})

    # (a) ANN classifier accuracy
    vals = [PAPER_TOP1, ours_top1(), OURS_TOP1_NO_ATT]
    bars = ax[0].bar([0, 1, 2], [100 * v for v in vals], width=0.7, edgecolor="white", linewidth=1.5,
                     color=[SERIES[0][2], SERIES[1][2], SERIES[1][2]])
    bars[2].set_hatch("///"); bars[2].set_alpha(0.75)
    bar_labels(ax[0], bars, "{:.1f} %")
    ax[0].set_xticks([0, 1, 2], ["Paper", "Ours\npitch/yaw ±5°", "Ours\npitch/yaw = 0"], fontsize=8)
    ax[0].set_ylabel("validation top-1 accuracy [%]"); ax[0].set_ylim(0, 75)
    ax[0].set_title("(a) ANN pose-label classifier", fontsize=10)

    # (b), (c) mean errors per PF configuration, plus LM refinement (the paper's future work)
    groups = [("PF N=1000\n(Fig. 9)", "fig9"), ("PF N=5000\n(Fig. 10)", "fig10"),
              ("LM refinement\n(future work)", "lm")]
    width = 0.2
    for k, (key, unit, title) in enumerate((("pos", "m", "(b) mean position error"),
                                             ("att", "°", "(c) mean attitude error"))):
        a = ax[k + 1]
        for s, (label, _, colour, tag) in enumerate(SERIES):
            means, stds = [], []
            for _, g in groups:
                if g == "lm":                         # LM rows: Gaussian runs with --refine
                    lm_tag = {"gauss_softprop": "gauss_refine", "gauss_sig1_softprop_los": "gauss_refine_los"}
                    if tag not in lm_tag:
                        means.append(np.nan); stds.append(0); continue
                    m = summary(lm_tag[tag])["ANN + LM refine"]
                elif tag is None:
                    _, pe, ae = load_paper(g)
                    means.append(np.nanmean(pe if key == "pos" else ae)); stds.append(0); continue
                else:
                    m = summary(tag)[CONFIGS[g][0]]
                means.append(m["pos_mean"] if key == "pos" else m["att_mean_deg"])
                stds.append(m["pos_mean_std"] if key == "pos" else m["att_mean_deg_std"])
            x = np.arange(len(groups)) + (s - 1.5) * width
            bars = a.bar(x, means, width, color=colour, edgecolor="white", linewidth=1.5,
                         label=label, yerr=stds, error_kw=dict(ecolor=MUTED, lw=0.8, capsize=2))
            bar_labels(a, bars, "{:.2f}", np.array(stds))
        a.text(2 - width, 0.15 * a.get_ylim()[1], "not done\nin paper", ha="center", fontsize=7, color=MUTED)
        a.set_xticks(range(len(groups)), [g[0] for g in groups], fontsize=8)
        a.set_ylabel(f"mean error over the 10 s approach [{'m' if key == 'pos' else 'deg'}]")
        a.set_title(title, fontsize=10)
    handles, labels = ax[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, 1.0))
    fig.text(0.01, 0.01, "Paper values: mean of the digitised Figs. 9/10 (single run). Ours: mean ± std "
             "over 10 Monte-Carlo runs. Hatched bar: side experiment with leader pitch/yaw fixed at 0.",
             fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.93)); fig.savefig(RESULTS_DIR / "paper_vs_ours_summary.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    for name in CONFIGS:
        time_series(name)
    summary_figure()
    print("saved results/paper_vs_ours_fig9.png, paper_vs_ours_fig10.png, paper_vs_ours_summary.png")
