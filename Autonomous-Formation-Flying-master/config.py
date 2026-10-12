"""Global configuration for the vision-based pose estimation project.

Reproduces: "Vision-Based Precision Pose Estimation For Autonomous Formation
Flying" (R. Punnoose, Stanford).

Conventions
-----------
* Frame F: follower-aligned frame, x forward, y right, z DOWN.
* A pose is a 6-vector  [x, y, z, phi, theta, psi]  (angles in RADIANS):
    (x, y, z)           = r_F/L, position of the FOLLOWER relative to the LEADER
                          (x = -90 means follower is 90 m behind the leader)
    (phi, theta, psi)   = leader roll, pitch, yaw relative to the follower
* The follower camera is gimballed so it always points at the leader.
"""
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
for _d in (DATA_DIR, MODEL_DIR, RESULTS_DIR):
    _d.mkdir(exist_ok=True)

# ---------------- Camera ----------------
IMG_W, IMG_H = 1280, 960
FOCAL = 1000.0                       # pixels
CX, CY = IMG_W / 2.0, IMG_H / 2.0
PIXEL_NOISE_STD = 0.5                # px, simulated CNN detector noise
PIXEL_NORM = 400.0                   # px, scaling of ANN inputs
LOS_NOISE_DEG = 0.1                  # deg, gimbal line-of-sight angle noise (--known-los extension)

# ---------------- Pose discretisation (paper Table I) ----------------
# name: (min, max, n_bins)      phi in DEGREES
POSE_BINS = {
    "x":   (-100.0, 50.0, 10),
    "y":   (-50.0, 50.0, 10),
    "z":   (-20.0, 20.0, 8),
    "phi": (-45.0, 45.0, 6),
}
BIN_SHAPE = tuple(v[2] for v in POSE_BINS.values())   # (10, 10, 8, 6)
N_LABELS = int(np.prod(BIN_SHAPE))                    # 4800
N_FEATURES = 14
N_INPUTS = 3 * N_FEATURES                              # 42

# ---------------- Sampling of training poses ----------------
MIN_RANGE = 10.0          # m, minimum 3D distance
MIN_HORIZ = 5.0           # m, avoid camera looking straight down
ATT_LIMIT_DEG = 5.0       # leader pitch / yaw sampled in +-5 deg (not part of label)

# ---------------- Dynamics ----------------
DT = 0.1                  # s  -> 100 steps in a 10 s scenario
V_NOM = 50.0              # m/s, nominal speed used to map pitch/roll -> velocity
LEADER_VX_DIST = 1.0      # m/s, uniform disturbance on leader forward velocity
LEADER_ANGLE_DIST_DEG = 0.5   # deg, leader pitch/roll command disturbance (-> vz, vy)
ATT_RW_DEG = 1.0          # deg per step, uniform random walk on leader attitude
