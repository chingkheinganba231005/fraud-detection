"""Figures for the README."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

from .scoring import LABELS  # noqa: E402

COLOURS = {"lightgbm": "#1b6ca8", "lightgbm + peer rule": "#7b3294", "logistic regression": "#e08214",
           "peer rule": "#4d9221", "isolation forest": "#8c8c8c", "nearest known cases": "#c51b7d"}


def _save(fig, path):
    path.parent.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def precision_recall(y, scores: dict, path):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for name, s in scores.items():
        p, r, _ = precision_recall_curve(y, s)
        ax.plot(r, p, label=name, color=COLOURS.get(name), lw=1.6)
    ax.axhline(y.mean(), color="black", lw=0.8, ls=":", label="random list")
    ax.set_yscale("log")
    ax.set_xlabel("share of later-excluded providers found (recall)")
    ax.set_ylabel("share of the list later excluded (precision)")
    ax.set_title("Held-out 2022 providers")
    ax.legend(fontsize=7, loc="upper right")
    _save(fig, path)


def recall_by_budget(results: dict, path):
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for name, r in results["detectors"].items():
        ks = [int(k) for k in r["budgets"]]
        rec = [r["budgets"][str(k)]["recall"] for k in ks]
        ax.plot(ks, rec, marker="o", label=name, color=COLOURS.get(name), lw=1.4)
        if name == results["chosen"]:
            lo = [r["recall_ci"][str(k)][0] for k in ks]
            hi = [r["recall_ci"][str(k)][1] for k in ks]
            ax.fill_between(ks, lo, hi, color=COLOURS.get(name), alpha=0.15)
    n = next(iter(results["detectors"].values()))["n"]
    ks = np.array(sorted(int(k) for k in next(iter(results["detectors"].values()))["budgets"]))
    ax.plot(ks, ks / n, color="black", ls=":", lw=0.8, label="random list")
    ax.set_xscale("log")
    ax.set_xlabel("providers reviewed (out of %s)" % f"{n:,}")
    ax.set_ylabel("later-excluded providers found")
    ax.legend(fontsize=7)
    _save(fig, path)


def importance(gain, path):
    names = [g[0] for g in gain][::-1]
    vals = [g[1] for g in gain][::-1]
    pretty = []
    for n in names:
        base = n[2:] if n.startswith("z_") else n
        label = LABELS.get(base, base.replace("_", " "))
        pretty.append(label + (" (vs peers)" if n.startswith("z_") else ""))
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.barh(pretty, vals, color="#1b6ca8")
    ax.set_xlabel("share of LightGBM gain")
    _save(fig, path)


def stress_heatmap(table: dict, detectors, path):
    schemes = list(table)
    strengths = sorted({float(s) for v in table.values() for s in v})
    fig, axes = plt.subplots(1, len(detectors), figsize=(3.2 * len(detectors), 3.0), sharey=True)
    for ax, det in zip(np.atleast_1d(axes), detectors):
        m = np.array([[table[sc][str(st)][det]["detected"] for st in strengths] for sc in schemes])
        ax.imshow(m, vmin=0, vmax=1, cmap="Blues", aspect="auto")
        for i in range(len(schemes)):
            for j in range(len(strengths)):
                ax.text(j, i, f"{m[i, j]:.0%}", ha="center", va="center", fontsize=7,
                        color="white" if m[i, j] > 0.6 else "black")
        ax.set_xticks(range(len(strengths)), [f"×{s:g}" for s in strengths], fontsize=7)
        ax.set_yticks(range(len(schemes)), schemes, fontsize=7)
        ax.set_title(det, fontsize=8)
    _save(fig, path)
