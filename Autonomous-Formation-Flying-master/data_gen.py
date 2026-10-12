"""Synthetic dataset generator (there is no public dataset for this paper).

Each sample: random relative pose within Table I ranges -> project 14 features
-> 42-d vector [x_px, y_px, occluded_flag] * 14  +  discrete pose label.

Usage:  python data_gen.py --n-train 400000 --n-val 50000
"""
import argparse
import numpy as np
from config import *
from aircraft import observe
from discretize import pose_to_label


def sample_poses(n, rng):
    out = np.empty((0, 6))
    lim = np.radians(ATT_LIMIT_DEG)
    while len(out) < n:
        m = int((n - len(out)) * 1.3) + 16
        x = rng.uniform(POSE_BINS["x"][0], POSE_BINS["x"][1], m)
        y = rng.uniform(POSE_BINS["y"][0], POSE_BINS["y"][1], m)
        z = rng.uniform(POSE_BINS["z"][0], POSE_BINS["z"][1], m)
        ok = (np.sqrt(x**2 + y**2 + z**2) >= MIN_RANGE) & (np.hypot(x, y) >= MIN_HORIZ)
        phi = np.radians(rng.uniform(POSE_BINS["phi"][0], POSE_BINS["phi"][1], m))
        th = rng.uniform(-lim, lim, m)
        ps = rng.uniform(-lim, lim, m)
        out = np.vstack([out, np.stack([x, y, z, phi, th, ps], 1)[ok]])
    return out[:n]


def features_to_input(uv, det):
    """(…,14,2) pixel coords + (…,14) detected mask -> (…,42) ANN input.
    Layout per feature: [x_norm, y_norm, occlusion_flag]; undetected -> x=y=0, flag=1."""
    xn = np.where(det, (uv[..., 0] - CX) / PIXEL_NORM, 0.0)
    yn = np.where(det, (uv[..., 1] - CY) / PIXEL_NORM, 0.0)
    flag = (~det).astype(float)
    return np.stack([xn, yn, flag], axis=-1).reshape(uv.shape[:-2] + (N_INPUTS,)).astype(np.float32)


def labels_from_poses(poses):
    p4 = np.column_stack([poses[:, :3], np.degrees(poses[:, 3])])
    return pose_to_label(p4)


def make_dataset(n, rng, chunk=50000):
    P, X = [], []
    for s in range(0, n, chunk):
        poses = sample_poses(min(chunk, n - s), rng)
        uv, det = observe(poses, rng, noise=True)
        X.append(features_to_input(uv, det)); P.append(poses)
    P = np.vstack(P); X = np.vstack(X)
    return X, labels_from_poses(P).astype(np.int32), P.astype(np.float32)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=400000)
    ap.add_argument("--n-val", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    for name, n in (("train", a.n_train), ("val", a.n_val)):
        X, y, P = make_dataset(n, rng)
        np.savez_compressed(DATA_DIR / f"{name}.npz", X=X, y=y, poses=P)
        print(f"{name}: X{X.shape} y{y.shape}  occluded fraction = {X[:, 2::3].mean():.3f}"
              f"  distinct labels = {len(np.unique(y))}")
