r"""Damped-least-squares inverse kinematics (Wampler 1986; Nakamura & Hanafusa 1986).

Task: end-effector position :math:`p(q) \in \mathbb{R}^3` plus a soft
"gripper points down" term. With hand axis :math:`a(q)` and desired
:math:`a^* = -\hat z`, the orientation error is :math:`e_o = a \times a^*`
(the small rotation that aligns them), whose Jacobian is the rotational
Jacobian :math:`J_r`. Stacking

.. math::
    e = \begin{bmatrix} p^* - p(q) \\ \beta\,(a \times a^*) \end{bmatrix},\qquad
    J = \begin{bmatrix} J_p \\ \beta J_r \end{bmatrix},

the DLS step minimises :math:`\lVert J\Delta q - e\rVert^2 + \lambda^2 \lVert\Delta q\rVert^2`:

.. math::
    \Delta q = J^\top \left(J J^\top + \lambda^2 I\right)^{-1} e,

which stays bounded near singularities (where the plain pseudo-inverse blows
up), at the cost of a small tracking bias :math:`O(\lambda^2)`. Iterate
:math:`q \leftarrow \mathrm{clip}(q + \alpha \Delta q)` within joint limits.
The arm has 4 joints for a 5-row task, so orientation is a soft secondary goal;
:math:`\beta < 1` favours position accuracy.
"""

from __future__ import annotations

import mujoco
import numpy as np

ARM_JOINTS = ("yaw", "shoulder", "elbow", "wrist")


class DLSIK:
    def __init__(self, model: mujoco.MjModel, site: str = "ee", damping: float = 0.05,
                 beta: float = 0.3, step: float = 1.0):
        self.m = model
        self.d = mujoco.MjData(model)          # scratch data, never stepped
        self.site = model.site(site).id
        self.qadr = np.array([model.joint(j).qposadr[0] for j in ARM_JOINTS])
        self.dadr = np.array([model.joint(j).dofadr[0] for j in ARM_JOINTS])
        self.lo = np.array([model.joint(j).range[0] for j in ARM_JOINTS])
        self.hi = np.array([model.joint(j).range[1] for j in ARM_JOINTS])
        self.lam, self.beta, self.alpha = damping, beta, step
        self._jp = np.zeros((3, model.nv))
        self._jr = np.zeros((3, model.nv))

    def fk(self, q: np.ndarray) -> np.ndarray:
        self.d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)
        return self.d.site_xpos[self.site].copy()

    def solve(self, target, q0, tol: float = 1e-4, max_iter: int = 300):
        """Return ``(q, position_error_m, iterations)``."""
        q = np.array(q0, float)
        a_star = np.array([0.0, 0.0, -1.0])
        for it in range(1, max_iter + 1):
            self.d.qpos[self.qadr] = q
            mujoco.mj_kinematics(self.m, self.d)
            mujoco.mj_comPos(self.m, self.d)
            p = self.d.site_xpos[self.site]
            a = self.d.site_xmat[self.site].reshape(3, 3)[:, 2]   # hand z-axis
            ep = target - p
            if np.linalg.norm(ep) < tol and it > 1:
                break
            mujoco.mj_jacSite(self.m, self.d, self._jp, self._jr, self.site)
            J = np.vstack([self._jp[:, self.dadr], self.beta * self._jr[:, self.dadr]])
            e = np.concatenate([ep, self.beta * np.cross(a, a_star)])
            dq = J.T @ np.linalg.solve(J @ J.T + self.lam ** 2 * np.eye(6), e)
            q = np.clip(q + self.alpha * dq, self.lo, self.hi)
        return q, float(np.linalg.norm(target - self.fk(q))), it
