"""EXTENSION (paper's suggested future work): non-linear least-squares refinement
of a coarse pose estimate by minimising feature reprojection error (PnP / bundle-
adjustment style).  Pitch and yaw are weakly observable under a gimballed camera
(they trade off against the bearing), so a weak Gaussian prior keeps them near 0."""
import numpy as np
from scipy.optimize import least_squares
from config import *
from aircraft import project


def refine_pose(pose0, uv_meas, det, prior_deg=ATT_LIMIT_DEG / 1.7, los=None):
    """`los` (optional, unit 3-vector): measured gimbal line of sight. When given, it adds
    a bearing residual that removes the bearing / leader-attitude ambiguity."""
    if det.sum() < 4:
        return pose0
    sig_px = PIXEL_NOISE_STD
    sig_ang = np.radians(prior_deg)
    sig_los = np.radians(LOS_NOISE_DEG)
    meas = uv_meas[det]

    def resid(p):
        pred = project(p[None])[0][det]
        r_pix = ((pred - meas) / sig_px).ravel()
        r_pri = p[4:] / sig_ang
        if los is None:
            return np.concatenate([r_pix, r_pri])
        r_los = np.cross(p[:3] / np.linalg.norm(p[:3]), los) / sig_los
        return np.concatenate([r_pix, r_pri, r_los])

    try:
        sol = least_squares(resid, pose0, method="trf", x_scale=np.array([5, 5, 5, .1, .05, .05]),
                            max_nfev=60)
        return sol.x
    except Exception:
        return pose0
