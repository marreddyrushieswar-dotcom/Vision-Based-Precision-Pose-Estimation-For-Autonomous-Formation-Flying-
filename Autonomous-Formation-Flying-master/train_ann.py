"""Train the ANN pose-label classifier and reproduce paper Figs. 4 and 5.

Usage:  python train_ann.py --epochs 20
        python train_ann.py --eval-only        # re-evaluate the saved model, no training
Outputs: models/ann.npz, models/error_pool.npy, results/fig4_*.png, results/fig5_*.png
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import *
import ann
from discretize import label_to_pose, pose_to_label, soft_pose


def hist_fig(err, title, path, log=False):
    names = ["x [m]", "y [m]", "z [m]", "phi [deg]"]
    fig, ax = plt.subplots(2, 2, figsize=(9, 6))
    for k, a in enumerate(ax.ravel()):
        a.hist(err[:, k], bins=60, color="#1f77b4", edgecolor="k", linewidth=0.2)
        a.set_xlabel(names[k]); a.set_ylabel("count")
        if log: a.set_yscale("log")
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-only", action="store_true", help="skip training, evaluate models/ann.npz")
    a = ap.parse_args()

    va = np.load(DATA_DIR / "val.npz")
    Xva, yva, Pva = va["X"], va["y"], va["poses"]
    if not a.eval_only:
        tr = np.load(DATA_DIR / "train.npz")
        Xtr, ytr = tr["X"], tr["y"]
        rng = np.random.default_rng(a.seed)
        params = ann.init_params(rng)
        print(f"train {Xtr.shape}, val {Xva.shape}, classes {N_LABELS}")
        hist = ann.train(params, Xtr, ytr, Xva, yva, a.epochs, a.batch, a.lr, rng=rng,
                         ckpt=MODEL_DIR / "ann.npz")
    params = ann.load()                           # best-validation checkpoint

    # ---- evaluation on validation set (batched: full 40k x 4800 logits would need ~2 GB) ----
    top5 = ann.predict_topk(params, Xva, k=5)
    pred = top5[:, 0]
    print(f"\nTop-1 accuracy: {(pred == yva).mean():.4f}   "
          f"Top-5 accuracy: {(top5 == yva[:, None]).any(1).mean():.4f}")

    true4 = np.column_stack([Pva[:, :3], np.degrees(Pva[:, 3])])
    cls4 = label_to_pose(pred)
    err_true = true4 - cls4                       # true pose - classified pose  (Fig. 4)
    err_lab = cls4 - label_to_pose(yva)           # classified pose - correct label (Fig. 5)
    np.save(MODEL_DIR / "error_pool.npy", err_true.astype(np.float32))
    hist_fig(err_true, "Error between classified pose and true pose", RESULTS_DIR / "fig4_true_vs_classified.png")
    hist_fig(err_lab, "Error between classified pose and correct label", RESULTS_DIR / "fig5_classified_vs_label.png", log=True)

    rms = np.sqrt((err_true ** 2).mean(0))
    print("RMS error true-vs-classified  x,y,z [m], phi [deg]:", np.round(rms, 2))
    wrong = pred != yva
    adj = (np.abs(err_lab[wrong]) <= np.array([15.0, 10.0, 5.0, 15.0]) + 1e-6).all(1).mean()
    print(f"Misclassified samples that land in an ADJACENT bin: {adj:.3f}")

    # extension: softmax-weighted mean of bin centres instead of the arg-max bin centre
    err_soft = true4 - soft_pose(ann.predict_proba(params, Xva))
    for nme, e in (("arg-max bin centre", err_true), ("softmax mean      ", err_soft)):
        print(f"{nme}: RMS x,y,z [m], phi [deg] = {np.round(np.sqrt((e ** 2).mean(0)), 2)}   "
              f"3-D position RMS = {np.sqrt((e[:, :3] ** 2).sum(1).mean()):.2f} m")
    if a.eval_only:
        raise SystemExit

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
    ax[0].plot([h[0] for h in hist]); ax[0].set_title("train loss"); ax[0].set_xlabel("epoch")
    ax[1].plot([h[1] for h in hist]); ax[1].set_title("validation accuracy"); ax[1].set_xlabel("epoch")
    fig.tight_layout(); fig.savefig(RESULTS_DIR / "ann_training_curves.png", dpi=130)
    print("saved models/ann.npz, models/error_pool.npy and figures in results/")
