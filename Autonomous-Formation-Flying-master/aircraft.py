"""Aircraft geometry, pinhole camera projection and occlusion model.

The aircraft is a simple F-16-like model: a fuselage ellipsoid + a thin wing
slab (used only for occlusion) and 14 point features at known locations in the
aircraft body frame (x forward, y right, z down).
"""
import numpy as np
from config import *

FEATURE_NAMES = [
    "nose", "L wingtip", "R wingtip", "L wing root TE", "R wing root TE",
    "fin tip", "L stabiliser tip", "R stabiliser tip", "exhaust", "canopy",
    "intake lip", "L wing LE", "R wing LE", "ventral fin",
]
FEATURES_BODY = np.array([
    [ 7.8,  0.0,  0.10],   # nose
    [-2.0, -5.0,  0.25],   # left wingtip
    [-2.0,  5.0,  0.25],   # right wingtip
    [-5.0, -1.2,  0.25],   # left wing root trailing edge
    [-5.0,  1.2,  0.25],   # right wing root trailing edge
    [-6.0,  0.0, -3.20],   # vertical fin tip (up = -z)
    [-8.0, -2.8,  0.30],   # left horizontal stabiliser tip
    [-8.0,  2.8,  0.30],   # right horizontal stabiliser tip
    [-8.3,  0.0,  0.10],   # engine exhaust
    [ 2.5,  0.0, -1.30],   # canopy top
    [ 3.0,  0.0,  1.10],   # intake lip (belly)
    [ 0.5, -3.0,  0.25],   # left wing leading edge
    [ 0.5,  3.0,  0.25],   # right wing leading edge
    [-5.0,  0.0,  1.50],   # ventral fin
])
assert FEATURES_BODY.shape == (N_FEATURES, 3)

# Occluders (body frame)
FUSELAGE_CENTER = np.array([-0.5, 0.0, 0.1])
FUSELAGE_AXES = np.array([7.5, 0.9, 0.9])          # ellipsoid semi-axes
WING_CENTER = np.array([-2.0, 0.0, 0.25])
WING_HALF = np.array([3.3, 4.0, 0.02])             # thin box half-extents
STANDOFF = 0.4                                      # m, ignore hits this close to a feature


def rot_matrix(phi, theta, psi):
    """Batched ZYX rotation  R = Rz(psi) Ry(theta) Rx(phi)  (body -> frame F)."""
    phi, theta, psi = np.atleast_1d(phi), np.atleast_1d(theta), np.atleast_1d(psi)
    cp, sp = np.cos(phi), np.sin(phi)
    ct, st = np.cos(theta), np.sin(theta)
    cy, sy = np.cos(psi), np.sin(psi)
    R = np.empty(phi.shape + (3, 3))
    R[..., 0, 0] = cy * ct
    R[..., 0, 1] = cy * st * sp - sy * cp
    R[..., 0, 2] = cy * st * cp + sy * sp
    R[..., 1, 0] = sy * ct
    R[..., 1, 1] = sy * st * sp + cy * cp
    R[..., 1, 2] = sy * st * cp - cy * sp
    R[..., 2, 0] = -st
    R[..., 2, 1] = ct * sp
    R[..., 2, 2] = ct * cp
    return R


def camera_rotation(r):
    """Gimballed camera: boresight (z_c) points at the leader, image 'down'
    is aligned with world down. Returns R_cam (N,3,3): v_cam = R_cam @ v_F."""
    r = np.atleast_2d(r)
    zc = -r / np.maximum(np.linalg.norm(r, axis=-1, keepdims=True), 1e-9)
    y_raw = np.array([0.0, 0.0, 1.0])[None, :] - zc[:, 2:3] * zc
    yc = y_raw / np.maximum(np.linalg.norm(y_raw, axis=-1, keepdims=True), 1e-6)
    xc = np.cross(yc, zc)
    return np.stack([xc, yc, zc], axis=1)


def project(poses):
    """Noise-free pinhole projection of the 14 features. poses: (N,6) -> uv (N,14,2)."""
    poses = np.atleast_2d(poses)
    r = poses[:, :3]
    R = rot_matrix(poses[:, 3], poses[:, 4], poses[:, 5])
    pts_F = -r[:, None, :] + np.einsum("nij,fj->nfi", R, FEATURES_BODY)
    pts_c = np.einsum("nij,nfj->nfi", camera_rotation(r), pts_F)
    z = np.maximum(pts_c[..., 2], 1e-3)
    u = FOCAL * pts_c[..., 0] / z + CX
    v = FOCAL * pts_c[..., 1] / z + CY
    return np.stack([u, v], axis=-1)


def occlusion_mask(poses):
    """True where a feature is hidden behind the fuselage or wing. (N,14) bool."""
    poses = np.atleast_2d(poses)
    r = poses[:, :3]
    R = rot_matrix(poses[:, 3], poses[:, 4], poses[:, 5])
    o = np.einsum("nji,nj->ni", R, r)                  # camera position in body frame
    d = FEATURES_BODY[None] - o[:, None, :]
    L = np.linalg.norm(d, axis=-1)
    tmax = np.clip(1.0 - STANDOFF / np.maximum(L, 1e-9), 0.0, 1.0)

    # ray - ellipsoid
    oc = (o - FUSELAGE_CENTER)[:, None, :] / FUSELAGE_AXES
    dd = d / FUSELAGE_AXES
    A = (dd ** 2).sum(-1)
    B = 2.0 * (oc * dd).sum(-1)
    C = (oc ** 2).sum(-1) - 1.0
    disc = B * B - 4 * A * C
    t1 = (-B - np.sqrt(np.maximum(disc, 0.0))) / (2 * A)
    ell = (disc > 0) & (t1 > 0) & (t1 < tmax)

    # ray - axis-aligned box (slab method)
    d_safe = np.where(np.abs(d) < 1e-9, 1e-9, d)
    ta = ((WING_CENTER - WING_HALF) - o[:, None, :]) / d_safe
    tb = ((WING_CENTER + WING_HALF) - o[:, None, :]) / d_safe
    tnear = np.minimum(ta, tb).max(-1)
    tfar = np.maximum(ta, tb).min(-1)
    box = (tnear < tfar) & (tnear > 0) & (tnear < tmax)
    return ell | box


def measure_los(r, rng, noise_deg=LOS_NOISE_DEG):
    """Gimbal line-of-sight measurement: unit vector along r_F/L (N,3) with small angular
    noise. The follower commands its own gimbal, so the pointing direction is known to it."""
    r = np.atleast_2d(r)
    u = r / np.linalg.norm(r, axis=1, keepdims=True)
    u = u + rng.normal(0.0, np.radians(noise_deg), u.shape)
    return u / np.linalg.norm(u, axis=1, keepdims=True)


def observe(poses, rng=None, noise=True):
    """Simulated CNN feature-detector output.
    Returns uv (N,14,2) pixel coordinates and det (N,14) bool = feature detected
    (not occluded and inside the image)."""
    uv = project(poses)
    occ = occlusion_mask(poses)
    if noise:
        rng = rng or np.random.default_rng()
        uv = uv + rng.normal(0.0, PIXEL_NOISE_STD, uv.shape)
    inside = (uv[..., 0] >= 0) & (uv[..., 0] < IMG_W) & (uv[..., 1] >= 0) & (uv[..., 1] < IMG_H)
    return uv, (~occ) & inside
