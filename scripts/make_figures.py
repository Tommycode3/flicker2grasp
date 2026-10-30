"""Figures + markdown tables from results/benchmark.csv, predictions.npz, robot.csv."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ssvep_robot.data import FREQS
from ssvep_robot.itr import itr

GAZE_SHIFT_S = 0.5
COLORS = {"CCA": "#2a78d6", "FBCCA": "#eb6834", "TRCA": "#1baf7a", "eTRCA": "#eda100"}
MARKERS = {"CCA": "o", "FBCCA": "s", "TRCA": "^", "eTRCA": "D"}
INK, MUTED = "#0b0b0b", "#52514e"
plt.rcParams.update({"axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.grid": True, "grid.color": "#e6e5e0",
                     "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 9, "lines.linewidth": 2})
FIG, RES = Path("figures"), Path("results")


def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def benchmark_figures():
    rows = [r for r in load(RES / "benchmark.csv") if r["task"] == "12class"]
    acc = defaultdict(dict)          # (method, window) -> {subject: acc}
    for r in rows:
        acc[(r["method"], float(r["window_s"]))][int(r["subject"])] = float(r["accuracy"])
    methods = ["CCA", "FBCCA", "TRCA", "eTRCA"]
    windows = sorted({w for _, w in acc})

    def stats(m, w, f):
        v = np.array([f(a, w) for a in acc[(m, w)].values()])
        return v.mean(), v.std(ddof=1) / np.sqrt(len(v))

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for m in methods:
        for ax, f in [(axes[0], lambda a, w: 100 * a),
                      (axes[1], lambda a, w: itr(12, a, w + GAZE_SHIFT_S))]:
            mu, se = np.array([stats(m, w, f) for w in windows]).T
            ax.errorbar(windows, mu, yerr=se, color=COLORS[m], marker=MARKERS[m], ms=5,
                        capsize=2, lw=2, label=m)
    axes[0].set(xlabel="window length [s]", ylabel="accuracy [%] (mean ± SEM, n=9)",
                ylim=(0, 102), title="12-class accuracy, leave-one-block-out")
    axes[0].axhline(100 / 12, color=MUTED, lw=1, ls=":")
    axes[0].text(3.95, 100 / 12 + 2, "chance", color=MUTED, ha="right")
    axes[1].set(xlabel="window length [s]", ylabel="ITR [bits/min] (mean ± SEM)",
                title=f"Wolpaw ITR, T = window + {GAZE_SHIFT_S} s gaze shift")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "accuracy_itr_vs_window.png", dpi=150)
    plt.close(fig)

    # per-subject figure at 1 s
    fig, ax = plt.subplots(figsize=(7, 3))
    subs = sorted(acc[("CCA", 1.0)])
    x = np.arange(len(subs))
    for i, m in enumerate(methods):
        ax.bar(x + (i - 1.5) * 0.2, [100 * acc[(m, 1.0)][s] for s in subs], 0.18,
               color=COLORS[m], label=m, edgecolor="white", linewidth=0.5)
    ax.set(xticks=x, xticklabels=[f"S{s}" for s in subs], ylabel="accuracy [%]",
           title="Per-subject accuracy at 1.0 s window")
    ax.legend(frameon=False, ncol=4, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "per_subject_1s.png", dpi=150)
    plt.close(fig)

    # markdown table
    sel = [0.5, 1.0, 2.0, 4.0]
    lines = ["| Method | " + " | ".join(f"Acc {w:g} s" for w in sel) + " | ITR max (window) |",
             "|---|" + "---|" * (len(sel) + 1)]
    for m in methods:
        cells = [f"{stats(m, w, lambda a, w: 100 * a)[0]:.1f} ± {stats(m, w, lambda a, w: 100 * a)[1]:.1f}"
                 for w in sel]
        itrs = [(stats(m, w, lambda a, w: itr(12, a, w + GAZE_SHIFT_S))[0], w) for w in windows]
        best = max(itrs)
        lines.append(f"| {m} | " + " | ".join(cells) + f" | {best[0]:.1f} bits/min ({best[1]:g} s) |")
    (RES / "table_benchmark.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def confusion_figures():
    npz = np.load(RES / "predictions.npz")
    order = np.argsort(FREQS)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    for ax, m in zip(axes, ["FBCCA", "eTRCA"]):
        C = np.zeros((12, 12))
        for s in range(1, 10):
            y, p = npz[f"12class|{m}|1.0|{s}|y"], npz[f"12class|{m}|1.0|{s}|pred"]
            np.add.at(C, (y, p), 1)
        C = C[np.ix_(order, order)]
        C = 100 * C / C.sum(axis=1, keepdims=True)
        im = ax.imshow(C, cmap="Blues", vmin=0, vmax=100)
        labels = [f"{f:g}" for f in FREQS[order]]
        ax.set(xticks=range(12), yticks=range(12), xlabel="decoded [Hz]", ylabel="true [Hz]",
               title=f"{m}, 1.0 s window")
        ax.set_xticklabels(labels, rotation=90)
        ax.set_yticklabels(labels)
        ax.grid(False)
        for i in range(12):
            ax.text(i, i, f"{C[i, i]:.0f}", ha="center", va="center", fontsize=6,
                    color="white" if C[i, i] > 60 else INK)
    fig.colorbar(im, ax=axes, shrink=0.8, label="% of trials of the true class (row-normalised)")
    fig.suptitle("Confusion matrices pooled over 9 subjects, leave-one-block-out", y=0.98)
    fig.savefig(FIG / "confusion_1s.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def robot_figures():
    p = RES / "robot.csv"
    if not p.exists():
        return
    rows = load(p)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for m in ["FBCCA", "eTRCA"]:
        rs = [r for r in rows if r["method"] == m]
        ws = sorted({float(r["window_s"]) for r in rs})
        sr = [np.mean([float(r["success_rate"]) for r in rs if float(r["window_s"]) == w]) for w in ws]
        axes[0].plot(ws, 100 * np.array(sr), color=COLORS[m], marker=MARKERS[m], ms=5, label=m)
        axes[1].scatter([100 * float(r["accuracy_8class"]) for r in rs],
                        [100 * float(r["success_rate"]) for r in rs],
                        color=COLORS[m], marker=MARKERS[m], s=22, label=m,
                        edgecolor="white", linewidth=0.6)
    axes[0].set(xlabel="window length [s]", ylabel="task success [%] (mean over 9 subjects)",
                ylim=(0, 102), title="Pick-and-place success vs window")
    axes[0].legend(frameon=False)
    axes[1].set(xlabel="8-class decoder accuracy [%] (one dot = subject × window)",
                ylabel="task success [%]", xlim=(0, 102), ylim=(-2, 102),
                title="Success is a steep function of accuracy")
    fig.tight_layout()
    fig.savefig(FIG / "robot_success.png", dpi=150)
    plt.close(fig)

    lines = ["| Method | Window | 8-class acc | ITR (8 targets) | Task success | Time to finish (successes) |",
             "|---|---|---|---|---|---|"]
    for m in ["FBCCA", "eTRCA"]:
        for w in sorted({float(r["window_s"]) for r in rows}):
            rs = [r for r in rows if r["method"] == m and float(r["window_s"]) == w]
            f = lambda k: np.nanmean([float(r[k]) for r in rs])  # noqa: E731
            lines.append(f"| {m} | {w:g} s | {100 * f('accuracy_8class'):.1f} % | "
                         f"{f('itr_8class'):.1f} bits/min | {100 * f('success_rate'):.1f} % | "
                         + (f"{t:.1f} s |" if np.isfinite(t := f("mean_time_if_success_s")) else "– |"))
    (RES / "table_robot.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    benchmark_figures()
    confusion_figures()
    robot_figures()
