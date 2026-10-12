"""Simplified relative-motion dynamics used by the particle filter and the truth simulator.

State  s = [x, y, z, phi, theta, psi]   (r_F/L and leader attitude relative to the follower)
Control u = [vx, theta, phi]            (forward relative velocity, pitch, roll)
  * forward velocity  : vx
  * pitch -> vertical : vz = -V_NOM * sin(theta)
  * roll  -> lateral  : vy =  V_NOM * sin(phi)
The leader's control is unknown to the follower, so it is sampled uniformly as a disturbance.
"""
import numpy as np
from config import *


def control_velocity(u):
    vx, th, ph = u
    return np.array([vx, V_NOM * np.sin(ph), -V_NOM * np.sin(th)])


def propagate(states, u_follower, rng, dt=DT):
    """Propagate N particles one step through the (disturbed) dynamics."""
    n = len(states)
    v_f = control_velocity(u_follower)[None, :]
    dist = np.radians(LEADER_ANGLE_DIST_DEG)
    v_l = np.stack([
        rng.uniform(-LEADER_VX_DIST, LEADER_VX_DIST, n),
        V_NOM * np.sin(rng.uniform(-dist, dist, n)),
        -V_NOM * np.sin(rng.uniform(-dist, dist, n)),
    ], axis=1)
    out = states.copy()
    out[:, :3] += dt * (v_f - v_l)
    rw = np.radians(ATT_RW_DEG)
    out[:, 3:] += rng.uniform(-rw, rw, (n, 3))
    return out
