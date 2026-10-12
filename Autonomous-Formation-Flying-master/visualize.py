"""Plot detected (green) vs occluded (red) features for a few example poses (cf. paper Figs. 2-3)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import *
from aircraft import observe, FEATURE_NAMES

poses = np.array([[-60, 20, -10, 0.3, 0.02, 0.0],
                  [-40, 10, 15, -0.4, 0.0, 0.03],
                  [-90, 25, 15, 0.17, 0.0, 0.0]], float)
uv, det = observe(poses, np.random.default_rng(0), noise=False)
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for i, a in enumerate(ax):
    for f in range(N_FEATURES):
        c = "g" if det[i, f] else "r"
        a.scatter(*uv[i, f], c=c, s=25)
        a.annotate(FEATURE_NAMES[f], uv[i, f], fontsize=6, xytext=(3, 3), textcoords="offset points")
    a.set_aspect("equal"); a.invert_yaxis(); a.set_title(f"pose {np.round(poses[i, :3]).astype(int)}  {det[i].sum()}/14 detected")
fig.tight_layout(); fig.savefig(RESULTS_DIR / "example_features.png", dpi=130)
print("saved results/example_features.png")
