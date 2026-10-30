import numpy as np
import pytest

from ssvep_robot.robot.ik import DLSIK
from ssvep_robot.robot.sim import Q_HOME_GUESS, ArmSim
from ssvep_robot.robot.task import (COMMANDS, TaskState, TrialBank, all_cells, cell_to_m,
                                    operator_intent, optimal_length, run_episode)


@pytest.fixture(scope="module")
def sim():
    return ArmSim()


def test_ik_converges_on_every_reachable_grid_cell(sim):
    errs = [sim.ik.solve(cell_to_m(c), Q_HOME_GUESS)[1] for c in all_cells()]
    assert max(errs) < 1e-3          # < 1 mm everywhere in the task workspace


def test_ik_converges_on_random_reachable_targets(sim):
    """Targets from forward kinematics of random joint configs are reachable by construction.

    Position-only DLS (beta=0): the soft "point down" term would legitimately trade
    position for orientation on targets that are reachable only with a tilted hand.
    """
    ik = DLSIK(sim.m, beta=0.0)
    rng = np.random.default_rng(0)
    for _ in range(30):
        q_true = rng.uniform([-1.5, 0.2, 0.5, 0.0], [1.5, 1.0, 2.0, 1.5])
        target = ik.fk(q_true)
        q, err, _ = ik.solve(target, Q_HOME_GUESS)
        assert err < 2e-3


def test_error_free_episode_succeeds_in_physics(sim):
    sim.reset()
    s = TaskState()
    track = []
    while not s.done:
        s.apply(operator_intent(s))
        track.append(sim.execute(s))
    assert s.success
    assert max(track) < 5e-3         # servos settle within 5 mm in 0.6 s


def test_perfect_decoder_uses_optimal_length():
    n = optimal_length()
    bank = TrialBank(np.repeat(np.arange(len(COMMANDS)), 5), np.repeat(np.arange(len(COMMANDS)), 5))
    s, log = run_episode(bank, np.random.default_rng(0), budget=2 * n)
    assert s.success and len(log) == n


def test_release_outside_bin_fails():
    s = TaskState()
    s.cell, s.carrying, s.closed = (2, 2, -2), True, True
    s.apply(7)
    assert s.done and not s.success
