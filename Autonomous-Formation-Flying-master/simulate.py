"""10 s formation-approach scenario: ANN-only vs ANN + particle filter (paper Figs. 6, 9, 10).

Usage:  python simulate.py                 # paper configs, averaged over 10 Monte-Carlo runs
        python simulate.py --likelihood gauss --refine
        python simulate.py --likelihood gauss --refine --known-los
        python simulate.py --seeds 1              # quick single run (live demo)
        python simulate.py --likelihood gauss --proposal softmax          # improved PF
        python simulate.py --likelihood gauss --proposal softmax --sigma-px 1 --refine --known-los
"""
import argparse, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import *
from aircraft import observe, rot_matrix, measure_los
from data_gen import features_to_input
from discretize import label_to_pose, soft_pose
from ann import ANNPredictor
from particle_filter import ParticleFilter
from refine import refine_pose


def build_trajectory(T=10.0, dt=DT):
    t = np.arange(0, T + 1e-9, dt)
    x = -90 + 5 * t                      # closing from -90 m to -40 m
    y = 25 - 1 * t
    z = np.full_like(t, 15.0)
    phi = np.radians(10 + 8 * np.sin(2 * np.pi * t / 6))
    th = np.radians(2 * np.sin(2 * np.pi * t / 4))
    ps = np.radians(2 * np.cos(2 * np.pi * t / 5))
    u_f = np.array([5.0, 0.0, np.arcsin(-1.0 / V_NOM)])    # follower control: vx, pitch, roll
    return t, np.stack([x, y, z, phi, th, ps], 1), u_f


def attitude_error_deg(est, true):
    Re = rot_matrix(est[:, 3], est[:, 4], est[:, 5])
    Rt = rot_matrix(true[:, 3], true[:, 4], true[:, 5])
    tr = np.einsum("nij,nij->n", Re, Rt)
    return np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1)))


def errors(est, true):
    return np.linalg.norm(est[:, :3] - true[:, :3], axis=1), attitude_error_deg(est, true)


CONFIGS = [("PF N=1000, a=0.9 every 10 cycles", 1000, 0.9, 10),
           ("PF N=5000, a=0.9 every cycle", 5000, 0.9, 1)]


def run_once(seed, a, predictor, pool, t, truth, u_f):
    """One Monte-Carlo run: fresh detector noise + fresh PF randomness. Returns {method: (K,6)}."""
    rng = np.random.default_rng(seed)
    K = len(t)
    uv, det = observe(truth, rng, noise=True)                  # simulated CNN detections
    los = measure_los(truth[:, :3], rng) if a.known_los else [None] * K
    probs = predictor.predict_proba(features_to_input(uv, det))   # (K,4800)
    lab4 = label_to_pose(probs.argmax(1))                      # (K,4) x,y,z,phi_deg

    def as_pose(p4):
        est = np.column_stack([p4[:, :3], np.radians(p4[:, 3]), np.zeros((K, 2))])
        if a.known_los:                                        # keep label range, use gimbal bearing
            est[:, :3] = np.linalg.norm(est[:, :3], axis=1, keepdims=True) * los
        return est

    results = {}
    ann_est = as_pose(lab4)
    results["ANN only"] = ann_est
    results["ANN only (softmax mean)"] = as_pose(soft_pose(probs))   # extension
    if a.refine:
        results["ANN + LM refine"] = np.array(
            [refine_pose(ann_est[k], uv[k], det[k], los=los[k]) for k in range(K)])

    for name, n, alpha, every in CONFIGS:
        pf = ParticleFilter(n, pool, alpha, every, a.likelihood, a.sigma_px, a.proposal, seed=seed)
        est = np.array([pf.step(u_f, uv[k], det[k], lab4[k], los[k], probs[k]) for k in range(K)])
        results[name] = est
        if a.refine:
            results[name + " + LM refine"] = np.array(
                [refine_pose(est[k], uv[k], det[k], los=los[k]) for k in range(K)])
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--likelihood", choices=["inv_mse", "gauss"], default="inv_mse")
    ap.add_argument("--sigma-px", type=float, default=5.0, help="gauss likelihood width [px]")
    ap.add_argument("--proposal", choices=["label", "softmax"], default="label",
                    help="PF new-particle source: paper (label + error pool) or ANN softmax (extension)")
    ap.add_argument("--refine", action="store_true", help="add LM reprojection refinement curves")
    ap.add_argument("--known-los", action="store_true",
                    help="extension: follower knows its gimbal line of sight (bearing to leader)")
    ap.add_argument("--model", default=str(MODEL_DIR / "ann.npz"), help="ANN weights (.npz)")
    ap.add_argument("--seeds", type=int, default=10, help="number of Monte-Carlo runs to average")
    ap.add_argument("--seed", type=int, default=1, help="first seed")
    a = ap.parse_args()

    t, truth, u_f = build_trajectory()
    predictor = ANNPredictor(a.model)
    pool = np.load(MODEL_DIR / "error_pool.npy")

    runs = []
    for i in range(a.seeds):
        runs.append(run_once(a.seed + i, a, predictor, pool, t, truth, u_f))
        print(f"run {i + 1}/{a.seeds} done", flush=True)
    results = runs[0]                                          # plots use the first run

    # ---- metrics: mean +- std over Monte-Carlo runs ----
    summary = {}
    print(f"\n{a.seeds} runs (mean +- std over runs)")
    print(f"{'method':42s} {'pos RMSE [m]':>15s} {'pos mean [m]':>15s} {'att mean [deg]':>15s}")
    for name in results:
        m = np.array([[np.sqrt((pe ** 2).mean()), pe.mean(), ae.mean()]
                      for pe, ae in (errors(r[name], truth) for r in runs)])
        mu, sd = m.mean(0), m.std(0)
        summary[name] = dict(pos_rmse=float(mu[0]), pos_rmse_std=float(sd[0]),
                             pos_mean=float(mu[1]), pos_mean_std=float(sd[1]),
                             att_mean_deg=float(mu[2]), att_mean_deg_std=float(sd[2]),
                             n_runs=a.seeds)
        print(f"{name:42s} {mu[0]:8.2f} +-{sd[0]:5.2f} {mu[1]:8.2f} +-{sd[1]:5.2f} "
              f"{mu[2]:8.2f} +-{sd[2]:5.2f}")
    tag = (f"_{a.likelihood}" + (f"_sig{a.sigma_px:g}" if a.likelihood == "gauss" and a.sigma_px != 5.0 else "")
           + ("_softprop" if a.proposal == "softmax" else "") + ("_refine" if a.refine else "")
           + ("_los" if a.known_los else "")
           + ("" if Path(a.model).name == "ann.npz" else "_" + Path(a.model).stem)
           + ("" if a.seeds == 10 else f"_{a.seeds}seeds"))   # demo runs don't overwrite results
    json.dump(summary, open(RESULTS_DIR / f"summary{tag}.json", "w"), indent=2)
    np.savez(RESULTS_DIR / f"estimates{tag}.npz", t=t, truth=truth,
             **{k.replace(" ", "_"): v for k, v in results.items()})

    # ---- plots (first run) ----
    plt.figure(figsize=(6, 3.5))
    for k, nme in enumerate(["x (forward)", "y (lateral)", "z (vertical)"]):
        plt.plot(t, truth[:, k], label=nme)
    plt.xlabel("time [s]"); plt.ylabel("relative position [m]"); plt.legend(); plt.grid(alpha=.3)
    plt.title("Fig. 6 - true relative position trajectory"); plt.tight_layout()
    plt.savefig(RESULTS_DIR / "fig6_trajectory.png", dpi=130); plt.close()

    def err_fig(names, title, fname):
        fig, ax = plt.subplots(2, 1, figsize=(7, 5.5), sharex=True)
        for nme in names:
            pe, ae = errors(results[nme], truth)
            ax[0].plot(t, pe, label=nme); ax[1].plot(t, ae, label=nme)
        ax[0].set_ylabel("position error [m]"); ax[1].set_ylabel("attitude error [deg]")
        ax[1].set_xlabel("time [s]"); ax[0].legend(fontsize=7); ax[0].set_title(title)
        for x in ax: x.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(RESULTS_DIR / fname, dpi=130); plt.close(fig)

    err_fig([CONFIGS[0][0], "ANN only"], "Fig. 9 equivalent: N=1000, a=0.9 every 10 cycles", f"fig9{tag}.png")
    err_fig([CONFIGS[1][0], "ANN only"], "Fig. 10 equivalent: N=5000, a=0.9 every cycle", f"fig10{tag}.png")
    err_fig(list(results.keys()), "All methods", f"comparison{tag}.png")
    print(f"figures saved in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
