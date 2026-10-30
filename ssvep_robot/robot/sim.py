"""MuJoCo execution of grid commands: IK -> position servos -> physics."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from .ik import DLSIK
from .task import CUBE_CELL, TaskState, all_cells, cell_to_m

XML = Path(__file__).with_name("arm.xml")
Q_HOME_GUESS = np.array([0.0, 0.3, 1.2, 1.0])


class ArmSim:
    """Physics-simulated arm. Each command is held for ``dt_cmd`` seconds."""

    def __init__(self, dt_cmd: float = 0.6):
        self.m = mujoco.MjModel.from_xml_path(str(XML))
        self.d = mujoco.MjData(self.m)
        self.ik = DLSIK(self.m)
        self.n_sub = int(round(dt_cmd / self.m.opt.timestep))
        self.cube_mocap = self.m.body("cube").mocapid[0]
        self.site = self.m.site("ee").id
        self._cache: dict = {}
        self.reset()

    def joint_target(self, cell) -> np.ndarray:
        """IK solution for a grid cell (cached; warm-started from home)."""
        if cell not in self._cache:
            q, err, _ = self.ik.solve(cell_to_m(cell), Q_HOME_GUESS)
            self._cache[cell] = (q, err)
        return self._cache[cell][0]

    def max_ik_error(self) -> float:
        return max(self.ik.solve(cell_to_m(c), Q_HOME_GUESS)[1] for c in all_cells())

    def reset(self) -> None:
        mujoco.mj_resetData(self.m, self.d)
        q = self.joint_target((0, 0, 0))
        self.d.qpos[self.ik.qadr] = q
        self.d.ctrl[:4] = q
        self.d.ctrl[4:] = 0.0
        self.d.mocap_pos[self.cube_mocap] = cell_to_m(CUBE_CELL)
        mujoco.mj_forward(self.m, self.d)

    def ee(self) -> np.ndarray:
        return self.d.site_xpos[self.site].copy()

    def execute(self, s: TaskState, n_chunks: int = 1, on_chunk=None) -> float:
        """Drive servos to the state's cell; return final Cartesian tracking error (m)."""
        self.d.ctrl[:4] = self.joint_target(s.cell)
        self.d.ctrl[4:] = -0.012 if s.closed else 0.008
        per = self.n_sub // n_chunks
        for _ in range(n_chunks):
            mujoco.mj_step(self.m, self.d, nstep=per)
            self.d.mocap_pos[self.cube_mocap] = self.ee() if s.carrying else s.cube_m
            if on_chunk is not None:
                on_chunk()
        return float(np.linalg.norm(self.ee() - cell_to_m(s.cell)))
