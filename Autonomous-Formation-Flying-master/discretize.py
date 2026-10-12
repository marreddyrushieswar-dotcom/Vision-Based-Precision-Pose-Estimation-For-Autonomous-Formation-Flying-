"""Pose <-> discrete label conversion (paper Table I: 10 x 10 x 8 x 6 = 4800)."""
import numpy as np
from config import POSE_BINS, BIN_SHAPE

_KEYS = ["x", "y", "z", "phi"]


def pose_to_label(p4):
    """p4: (N,4) array [x, y, z, phi_deg] -> integer labels (N,)."""
    p4 = np.atleast_2d(p4)
    idx = []
    for k, key in enumerate(_KEYS):
        lo, hi, n = POSE_BINS[key]
        i = np.floor((p4[:, k] - lo) / ((hi - lo) / n)).astype(int)
        idx.append(np.clip(i, 0, n - 1))
    return np.ravel_multi_index(idx, BIN_SHAPE)


def label_to_pose(labels):
    """labels (N,) -> bin-centre poses (N,4) [x, y, z, phi_deg]."""
    idx = np.unravel_index(np.atleast_1d(labels), BIN_SHAPE)
    out = np.empty((len(idx[0]), 4))
    for k, key in enumerate(_KEYS):
        lo, hi, n = POSE_BINS[key]
        out[:, k] = lo + (idx[k] + 0.5) * (hi - lo) / n
    return out


BIN_CENTERS = label_to_pose(np.arange(np.prod(BIN_SHAPE)))                 # (4800,4)
BIN_WIDTHS = np.array([(POSE_BINS[k][1] - POSE_BINS[k][0]) / POSE_BINS[k][2] for k in _KEYS])


def soft_pose(probs):
    """EXTENSION: softmax-weighted mean of the bin centres, (N,4800) -> (N,4) [x, y, z, phi_deg].
    A continuous estimate that interpolates between neighbouring bins instead of snapping
    to the arg-max bin centre."""
    return probs @ BIN_CENTERS
