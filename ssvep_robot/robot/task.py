"""Pick-and-place task driven by decoded SSVEP commands.

Command alphabet (8 SSVEP targets -> 8 commands). We use the first 8 targets
of the 12-target tensor; the BCI is decoded as an 8-class problem, so the
decoder can only confuse real commands with real commands (no "free"
idle targets that would silently absorb errors).

The end-effector moves on a 5 cm Cartesian grid ("jog"). The operator is a
simulated *closed-loop* user: at each selection they look at the arm and pick
the command that best advances the task from the *current* state, so a wrong
jog is later corrected by the opposite jog, exactly as a real user would.
What the arm actually executes is the command decoded from a **real recorded
EEG trial** of the intended class (held-out block, LOBO).

Failure modes: releasing the cube outside the bin drops it (episode fails);
exceeding the command budget (2x the error-free command count) fails.
Grasping is a kinematic attachment (cube becomes a child of the gripper when
the fingertip is within 3 cm) - no contact-rich grasp physics.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

ROBOT_CLASSES = np.arange(8)       # indices into data.FREQS
COMMANDS = ["+X", "-X", "+Y", "-Y", "+Z", "-Z", "GRASP", "RELEASE"]
JOG = {0: (1, 0, 0), 1: (-1, 0, 0), 2: (0, 1, 0), 3: (0, -1, 0), 4: (0, 0, 1), 5: (0, 0, -1)}
GRASP, RELEASE = 6, 7

STEP_M = 0.05
HOME_M = np.array([0.30, 0.0, 0.20])
CUBE_CELL = (2, 2, -3)             # (0.40, 0.10, 0.05) m
BIN_CELL = (0, -4, -2)             # release point above bin centre (0.30, -0.20, 0.10) m
BIN_HALF_M = 0.07
SAFE_Z = -2                        # carry height (cells) to clear the bin wall
LO = np.array([-3, -5, -3])
HI = np.array([3, 4, 2])


def cell_to_m(cell) -> np.ndarray:
    return HOME_M + STEP_M * np.asarray(cell, float)


def all_cells():
    for x in range(LO[0], HI[0] + 1):
        for y in range(LO[1], HI[1] + 1):
            for z in range(LO[2], HI[2] + 1):
                yield (x, y, z)


@dataclass
class TaskState:
    cell: tuple = (0, 0, 0)
    closed: bool = False
    carrying: bool = False
    cube_m: np.ndarray = field(default_factory=lambda: cell_to_m(CUBE_CELL))
    done: bool = False
    success: bool = False
    reason: str = ""

    def apply(self, cmd: int) -> None:
        """Execute a (decoded) command on the grid state."""
        if cmd in JOG:
            c = np.clip(np.add(self.cell, JOG[cmd]), LO, HI)
            self.cell = tuple(int(v) for v in c)
        elif cmd == GRASP:
            self.closed = True
            if np.linalg.norm(cell_to_m(self.cell) - self.cube_m) < 0.03 and not self.carrying:
                self.carrying = True
        elif cmd == RELEASE:
            self.closed = False
            if self.carrying:
                self.carrying = False
                p = cell_to_m(self.cell)
                in_bin = np.all(np.abs(p[:2] - cell_to_m(BIN_CELL)[:2]) < BIN_HALF_M)
                self.cube_m = np.array([p[0], p[1], 0.02 + (0.008 if in_bin else 0.0)])
                self.done, self.success = True, bool(in_bin)
                self.reason = "placed in bin" if in_bin else "dropped outside bin"
        if self.carrying:
            self.cube_m = cell_to_m(self.cell)


def operator_intent(s: TaskState) -> int:
    """Closed-loop user policy: the command that best advances the task now."""
    if not s.carrying:
        if s.closed:
            return RELEASE                       # open an empty gripper first
        goal = CUBE_CELL
        if s.cell == goal:
            return GRASP
        order = (0, 1, 2)                        # x, y, then descend
    else:
        if s.cell[2] < SAFE_Z:
            return 4                             # lift before moving sideways
        goal = BIN_CELL
        if s.cell == goal:
            return RELEASE
        order = (0, 1, 2)
    for ax in order:
        d = goal[ax] - s.cell[ax]
        if d:
            return [0, 2, 4][ax] + (0 if d > 0 else 1)
    raise RuntimeError("unreachable")


def optimal_length() -> int:
    s, n = TaskState(), 0
    while not s.done:
        s.apply(operator_intent(s))
        n += 1
    return n


class TrialBank:
    """Real out-of-sample decoder outputs for one subject, grouped by true class."""

    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray):
        self.by_class = {c: y_pred[y_true == c] for c in np.unique(y_true)}

    def decode(self, intended: int, rng: np.random.Generator) -> int:
        return int(rng.choice(self.by_class[intended]))


def run_episode(bank: TrialBank, rng: np.random.Generator, budget: int, on_step=None):
    """Run one episode. Returns ``(state, log)`` with ``log = [(intended, executed), ...]``."""
    s, log = TaskState(), []
    while not s.done and len(log) < budget:
        intended = operator_intent(s)
        executed = bank.decode(intended, rng)
        s.apply(executed)
        log.append((intended, executed))
        if on_step is not None:
            on_step(s, intended, executed)
    if not s.done:
        s.reason = "command budget exhausted"
    return s, log
