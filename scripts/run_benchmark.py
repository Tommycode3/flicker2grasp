"""Benchmark CCA / FBCCA / TRCA / eTRCA on Nakanishi2015 with LOBO cross-validation.

Low-RAM by design: one subject in memory at a time, single process.
Outputs: results/benchmark.csv (per subject/method/window accuracy) and
results/predictions.npz (every out-of-sample prediction, used by the robot demo).
"""

from __future__ import annotations

import argparse
import csv
import gc
import time
from pathlib import Path

import numpy as np

from ssvep_robot.data import N_SUBJECTS, load_subject
from ssvep_robot.evaluate import METHODS, lobo_predict
from ssvep_robot.robot.task import ROBOT_CLASSES

WINDOWS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
ROBOT_WINDOWS = [0.5, 0.75, 1.0, 1.5, 2.0]
ROBOT_METHODS = ("FBCCA", "eTRCA")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, nargs="*", default=list(range(1, N_SUBJECTS + 1)))
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()
    args.out.mkdir(exist_ok=True)

    rows, preds = [], {}
    for s in args.subjects:
        t0 = time.time()
        sd = load_subject(s)
        for tag, classes, methods, windows in [
            ("12class", None, METHODS, WINDOWS),
            ("robot8", ROBOT_CLASSES, ROBOT_METHODS, ROBOT_WINDOWS),
        ]:
            for m in methods:
                for w in windows:
                    y, p, b = lobo_predict(sd, m, w, classes)
                    rows.append(dict(task=tag, subject=s, method=m, window_s=w,
                                     n_classes=int(y.max() + 1), n_trials=len(y),
                                     accuracy=float((y == p).mean())))
                    key = f"{tag}|{m}|{w}|{s}"
                    preds[key + "|y"], preds[key + "|pred"], preds[key + "|block"] = y, p, b
        del sd
        gc.collect()
        acc = [r["accuracy"] for r in rows if r["subject"] == s and r["task"] == "12class"
               and r["window_s"] == 1.0]
        print(f"S{s}: {time.time() - t0:5.1f}s  acc@1s CCA/FBCCA/TRCA/eTRCA = "
              + " ".join(f"{a:.3f}" for a in acc), flush=True)

    with open(args.out / "benchmark.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    np.savez_compressed(args.out / "predictions.npz", **preds)


if __name__ == "__main__":
    main()
