"""Pose-label classifier: pure-NumPy MLP (42 -> 100 -> 100 -> 4800, ReLU, softmax)
trained with Adam, exactly as described in the paper (no PyTorch needed)."""
import time
import numpy as np
from config import *


def init_params(rng, sizes=(N_INPUTS, 100, 100, N_LABELS)):
    p = {}
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:]), 1):
        p[f"W{i}"] = (rng.standard_normal((a, b)) * np.sqrt(2.0 / a)).astype(np.float32)
        p[f"b{i}"] = np.zeros(b, np.float32)
    return p


def forward(p, X):
    z1 = X @ p["W1"] + p["b1"]; h1 = np.maximum(z1, 0)
    z2 = h1 @ p["W2"] + p["b2"]; h2 = np.maximum(z2, 0)
    logits = h2 @ p["W3"] + p["b3"]
    return logits, (X, z1, h1, z2, h2)


def loss_and_grads(p, X, y):
    logits, (X, z1, h1, z2, h2) = forward(p, X)
    logits = logits - logits.max(1, keepdims=True)
    e = np.exp(np.maximum(logits, -80.0))
    probs = e / e.sum(1, keepdims=True)
    n = len(y)
    loss = -np.log(probs[np.arange(n), y] + 1e-12).mean()
    # Flush negligible probabilities to 0: once the net is confident, many of the 4800
    # softmax outputs become float32 *denormals*, which makes the matmuls below ~6x slower.
    d3 = probs; d3[d3 < 1e-30] = 0.0; d3[np.arange(n), y] -= 1.0; d3 /= n
    g = {"W3": h2.T @ d3, "b3": d3.sum(0)}
    d2 = (d3 @ p["W3"].T) * (z2 > 0)
    g["W2"] = h1.T @ d2; g["b2"] = d2.sum(0)
    d1 = (d2 @ p["W2"].T) * (z1 > 0)
    g["W1"] = X.T @ d1; g["b1"] = d1.sum(0)
    return loss, g


class Adam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps, self.t = lr, b1, b2, eps, 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params, grads, lr=None):
        self.t += 1
        lr = lr or self.lr
        for k in params:
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * grads[k]
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * grads[k] ** 2
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= (lr * mh / (np.sqrt(vh) + self.eps)).astype(np.float32)


def predict(p, X, batch=8192):
    out = []
    for s in range(0, len(X), batch):
        out.append(forward(p, X[s:s + batch])[0].argmax(1))
    return np.concatenate(out)


def predict_topk(p, X, k=5, batch=4096):
    """Top-k labels per sample (column 0 = top-1), evaluated in batches to bound memory."""
    out = []
    for s in range(0, len(X), batch):
        logits = forward(p, X[s:s + batch])[0]
        top = np.argpartition(-logits, k, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(logits, top, 1), axis=1)
        out.append(np.take_along_axis(top, order, 1))
    return np.concatenate(out)


def predict_proba(p, X, batch=4096):
    """Softmax class probabilities (N, 4800), evaluated in batches."""
    out = []
    for s in range(0, len(X), batch):
        logits = forward(p, X[s:s + batch])[0].astype(np.float64)
        e = np.exp(logits - logits.max(1, keepdims=True))
        out.append(e / e.sum(1, keepdims=True))
    return np.concatenate(out)


def accuracy(p, X, y):
    return float((predict(p, X) == y).mean())


def train(p, Xtr, ytr, Xva, yva, epochs=20, batch=512, lr=1e-3, decay=0.92, rng=None,
          ckpt=None):
    """Mini-batch Adam training. If `ckpt` is a path, the best-validation weights are
    saved there after every epoch (so an interrupted run still leaves a usable model)."""
    rng = rng or np.random.default_rng(0)
    opt = Adam(p, lr)
    hist, best = [], -1.0
    for ep in range(epochs):
        t0 = time.time()
        perm = rng.permutation(len(Xtr))
        tot = 0.0
        for s in range(0, len(perm), batch):
            idx = perm[s:s + batch]
            loss, g = loss_and_grads(p, Xtr[idx], ytr[idx])
            opt.step(p, g, lr * decay ** ep)
            tot += loss * len(idx)
        va = accuracy(p, Xva, yva)
        hist.append((tot / len(perm), va))
        print(f"epoch {ep + 1:3d}/{epochs}  train loss {tot / len(perm):.4f}  "
              f"val acc {va:.4f}  ({time.time() - t0:.1f}s)", flush=True)
        if ckpt is not None and va > best:
            best = va
            save(p, ckpt)
    return hist


def save(p, path=MODEL_DIR / "ann.npz"):
    np.savez(path, **p)


def load(path=MODEL_DIR / "ann.npz"):
    d = np.load(path)
    return {k: d[k] for k in d.files}


class ANNPredictor:
    def __init__(self, path=MODEL_DIR / "ann.npz"):
        self.p = load(path)

    def predict(self, X):
        return predict(self.p, np.atleast_2d(X).astype(np.float32))

    def predict_proba(self, X):
        return predict_proba(self.p, np.atleast_2d(X).astype(np.float32))
