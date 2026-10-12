"""Time-parameterise a path under speed, acceleration and turning limits.

Speed at each point is capped by v_max and by the curvature limit
sqrt(a_lat_max / |kappa|) so the robot slows for tight turns. A forward pass
applies the acceleration limit and a backward pass the deceleration limit, so
the profile starts and ends at rest.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class Trajectory:
    xy: np.ndarray      # (N, 2) metres
    s: np.ndarray       # arc length at each point
    v: np.ndarray       # planned speed
    t: np.ndarray       # planned time stamp
    kappa: np.ndarray   # curvature, 1/m

    @property
    def duration(self):
        return float(self.t[-1])

    @property
    def length(self):
        return float(self.s[-1])

    def speed_at(self, s):
        return float(np.interp(s, self.s, self.v))

    def position_at_time(self, t):
        return np.array([np.interp(t, self.t, self.xy[:, 0]), np.interp(t, self.t, self.xy[:, 1])])


def curvature(xy):
    """Menger curvature through each point and its neighbours (0 at the ends)."""
    k = np.zeros(len(xy))
    if len(xy) < 3:
        return k
    a, b, c = xy[:-2], xy[1:-1], xy[2:]
    ab = np.linalg.norm(b - a, axis=1)
    bc = np.linalg.norm(c - b, axis=1)
    ca = np.linalg.norm(a - c, axis=1)
    cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    denom = ab * bc * ca
    with np.errstate(divide="ignore", invalid="ignore"):
        k[1:-1] = np.where(denom > 1e-12, 2 * cross / denom, 0.0)
    return k


def build(xy, v_max, a_max, a_lat_max, smooth_window=9):
    xy = np.asarray(xy, dtype=float)
    ds = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(ds)])

    kappa = curvature(xy)
    # curvature from a densified polyline is spiky; a short moving max keeps the
    # slowdown in place for the whole turn instead of a single point
    k_abs = np.abs(kappa)
    if smooth_window > 1 and len(k_abs) > smooth_window:
        pad = smooth_window // 2
        padded = np.pad(k_abs, pad, mode="edge")
        k_abs = np.array([padded[i:i + smooth_window].max() for i in range(len(k_abs))])

    with np.errstate(divide="ignore"):
        v_curve = np.where(k_abs > 1e-9, np.sqrt(a_lat_max / k_abs), np.inf)
    v_lim = np.minimum(v_max, v_curve)
    v_lim[0] = v_lim[-1] = 0.0

    v = v_lim.copy()
    for i in range(len(v) - 1):  # accelerate
        v[i + 1] = min(v[i + 1], np.sqrt(v[i] ** 2 + 2 * a_max * ds[i]))
    for i in range(len(v) - 2, -1, -1):  # decelerate
        v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2 * a_max * ds[i]))

    v_avg = 0.5 * (v[:-1] + v[1:])
    with np.errstate(divide="ignore"):
        # from rest the first segment would be ds / 0; use the constant-accel time instead
        dt = np.where(v_avg > 1e-9, ds / np.maximum(v_avg, 1e-9), np.sqrt(2 * ds / a_max))
    t = np.concatenate([[0.0], np.cumsum(dt)])
    return Trajectory(xy, s, v, t, kappa)


def accelerations(traj):
    dv = np.diff(traj.v)
    dt = np.diff(traj.t)
    return dv / np.maximum(dt, 1e-9)
