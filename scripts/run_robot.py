"""Replay real held-out SSVEP trials -> decoded commands -> MuJoCo arm.

For every (method, window, subject) we run N closed-loop pick-and-place
episodes. Each selection draws a *real* recorded trial of the intended class
from that subject and executes the class the decoder assigned to it in
leave-one-block-out evaluation (results/predictions.npz, task "robot8").
Every executed command is played out in MuJoCo physics (IK -> position servos).

Outputs: results/robot.csv, figures/robot_*.png, figures/robot_episode.gif.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

if sys.platform == "darwin":
    os.environ.setdefault("MUJOCO_GL", "cgl")
else:
    os.environ.setdefault("MUJOCO_GL", "egl")

from ssvep_robot.itr import itr  # noqa: E402
from ssvep_robot.robot.sim import ArmSim  # noqa: E402
from ssvep_robot.robot.task import (COMMANDS, TaskState, TrialBank, operator_intent,  # noqa: E402
                                    optimal_length)

GAZE_SHIFT_S = 0.5


def load_banks(npz, method, window):
    out = {}
    for s in range(1, 10):
        key = f"robot8|{method}|{window}|{s}"
        if key + "|y" in npz:
            out[s] = (npz[key + "|y"], npz[key + "|pred"])
    return out


def run_physics_episode(sim: ArmSim, bank: TrialBank, rng, budget, render=None):
    sim.reset()
    s, log, track = TaskState(), [], []
    while not s.done and len(log) < budget:
        intended = operator_intent(s)
        executed = bank.decode(intended, rng)
        s.apply(executed)
        log.append((intended, executed))
        if render is None:
            track.append(sim.execute(s))
        else:
            track.append(sim.execute(s, n_chunks=6, on_chunk=lambda: render(s, intended, executed, len(log))))
    if not s.done:
        s.reason = "command budget exhausted"
    return s, log, track


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=50, help="episodes per subject/condition")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-gif", action="store_true")
    ap.add_argument("--gif-only", action="store_true", help="skip the Monte-Carlo, only render")
    args = ap.parse_args()

    npz = np.load("results/predictions.npz")
    n_opt = optimal_length()
    budget = 2 * n_opt
    sim = ArmSim()
    if args.gif_only:
        make_gif(sim, npz, n_opt, budget, args.seed)
        return
    rows = []
    for method in ("FBCCA", "eTRCA"):
        for window in (0.5, 0.75, 1.0, 1.5, 2.0):
            banks = load_banks(npz, method, window)
            for subj, (y, p) in banks.items():
                bank = TrialBank(y, p)
                rng = np.random.default_rng([args.seed, subj, int(window * 100), method == "eTRCA"])
                succ, ncmd, track_max, reasons = [], [], 0.0, {}
                for _ in range(args.episodes):
                    s, log, track = run_physics_episode(sim, bank, rng, budget)
                    succ.append(s.success)
                    reasons[s.reason] = reasons.get(s.reason, 0) + 1
                    if s.success:
                        ncmd.append(len(log))
                    track_max = max(track_max, max(track))
                acc = float((y == p).mean())
                rows.append(dict(
                    method=method, window_s=window, subject=subj, accuracy_8class=acc,
                    itr_8class=itr(8, acc, window + GAZE_SHIFT_S),
                    success_rate=float(np.mean(succ)),
                    mean_cmds_if_success=float(np.mean(ncmd)) if ncmd else float("nan"),
                    mean_time_if_success_s=(float(np.mean(ncmd)) * (window + GAZE_SHIFT_S)
                                            if ncmd else float("nan")),
                    dropped=reasons.get("dropped outside bin", 0),
                    budget_exhausted=reasons.get("command budget exhausted", 0),
                    max_tracking_err_mm=1000 * track_max))
                print(f"{method:5s} {window:4.2f}s S{subj}: acc={acc:.3f} "
                      f"success={np.mean(succ):.2f} {reasons}", flush=True)

    Path("results").mkdir(exist_ok=True)
    with open("results/robot.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(f"optimal command count = {n_opt}, budget = {budget}")

    if not args.no_gif:
        make_gif(sim, npz, n_opt, budget, args.seed)


def make_gif(sim: ArmSim, npz, n_opt, budget, seed, method="eTRCA", window=0.5):
    """Render one real-trial episode. Picks the subject/seed whose successful episode
    contains at least one decoding error, so the GIF shows an error being corrected."""
    import imageio.v2 as imageio
    import matplotlib.pyplot as plt
    import mujoco

    banks = load_banks(npz, method, window)
    chosen = None
    for subj in sorted(banks, key=lambda s: (banks[s][0] == banks[s][1]).mean()):
        for ep_seed in range(200):
            rng = np.random.default_rng([seed, 999, subj, ep_seed])
            s, log, _ = run_physics_episode(sim, TrialBank(*banks[subj]), rng, budget)
            if s.success and any(i != e for i, e in log):
                chosen = (subj, ep_seed)
                break
        if chosen:
            break
    subj, ep_seed = chosen

    renderer = mujoco.Renderer(sim.m, 270, 360)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.25, -0.03, 0.12]
    cam.distance, cam.azimuth, cam.elevation = 1.05, 135, -25
    frames, ee_path, labels = [], [], []

    def render(s, intended, executed, k):
        renderer.update_scene(sim.d, camera=cam)
        img = renderer.render()
        frames.append((img, k, intended, executed))
        ee_path.append(sim.ee())
        labels.append(intended == executed)

    rng = np.random.default_rng([seed, 999, subj, ep_seed])
    s, log, _ = run_physics_episode(sim, TrialBank(*banks[subj]), rng, budget, render=render)

    # caption strip under each frame (matplotlib text, no extra font deps)
    out = []
    fig = plt.figure(figsize=(4.8, 4.2), dpi=75)
    for img, k, i, e in frames:
        fig.clf()
        ax = fig.add_axes([0, 0.14, 1, 0.86])
        ax.imshow(img)
        ax.axis("off")
        ok = i == e
        fig.text(0.02, 0.085, f"S{subj} | {method} | {window} s window | selection {k}/{len(log)}",
                 color="#52514e", fontsize=8)
        fig.text(0.02, 0.025, f"intended {COMMANDS[i]:<8s} decoded {COMMANDS[e]:<8s}"
                 + ("" if ok else "  <- decoding error"),
                 color=("#0b0b0b" if ok else "#c62828"), fontsize=8, family="monospace")
        fig.canvas.draw()
        out.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    plt.close(fig)
    Path("figures").mkdir(exist_ok=True)
    imageio.mimsave("figures/robot_episode.gif", out[::2], duration=0.12, loop=0)

    # static trajectory figure: 3-D path + per-axis time course
    P = np.array(ee_path)
    t = np.arange(len(P)) * sim.n_sub // 6 * sim.m.opt.timestep
    sel_end = P[5::6]
    errs = [n for n, (i, e) in enumerate(log) if i != e]
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.plot(P[:, 0], P[:, 1], P[:, 2], color="0.5", lw=1)
    ax.scatter(*sel_end.T, color="#2a78d6", s=14, label="after each selection")
    if errs:
        ax.scatter(*sel_end[errs].T, color="#c62828", s=60, marker="x", lw=2, zorder=5,
                   label="after a decoding error")
    ax.scatter(0.40, 0.10, 0.05, marker="s", s=80, color="#eb6834", label="cube start")
    ax.scatter(0.30, -0.20, 0.02, marker="s", s=80, color="#1baf7a", label="bin")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.legend(fontsize=7, loc="upper left")
    ax2 = fig.add_subplot(1, 2, 2)
    for j, (lab, c) in enumerate(zip("xyz", ["#2a78d6", "#eb6834", "#1baf7a"])):
        ax2.plot(t, P[:, j], color=c, lw=2, label=lab)
    sel_t = t[5::6]
    for n in errs:
        ax2.axvspan(sel_t[n] - sim.n_sub * sim.m.opt.timestep, sel_t[n], color="#c62828", alpha=0.15, lw=0)
    ax2.set(xlabel="simulated time [s] (0.6 s per selection)", ylabel="end-effector position [m]")
    ax2.legend(frameon=False, ncol=3)
    fig.suptitle(f"S{subj}, {method} {window} s: {len(log)} selections ({n_opt} optimal), "
                 f"{len(errs)} decoding error(s) (red), outcome: {s.reason}", fontsize=9)
    fig.tight_layout()
    fig.savefig("figures/robot_trajectory.png", dpi=130)
    plt.close(fig)
    print(f"GIF: subject {subj}, {len(log)} selections ({n_opt} optimal), "
          f"{len(errs)} errors, outcome: {s.reason}")


if __name__ == "__main__":
    main()
