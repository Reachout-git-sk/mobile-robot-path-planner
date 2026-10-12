"""Wheel speed step response and a simple grid search over PI gains."""
import itertools

import numpy as np

from .models import Motor
from .pid import PID, PIDGains


def step_response(gains, setpoint=1.0, duration=3.0, dt=0.02, tau=0.15, motor_gain=0.9, noise=0.0, seed=0):
    motor = Motor(tau=tau, gain=motor_gain, noise_std=noise, rng=np.random.default_rng(seed))
    pid = PID(gains, -1.5, 1.5)
    t = np.arange(0, duration, dt)
    y = np.empty_like(t)
    u = np.empty_like(t)
    for i in range(len(t)):
        u[i] = pid.update(setpoint, motor.speed, dt)
        y[i] = motor.step(u[i], dt)

    peak = y.max()
    overshoot = max(0.0, (peak - setpoint) / setpoint * 100)
    band = 0.02 * abs(setpoint)
    outside = np.nonzero(np.abs(y - setpoint) > band)[0]
    settling = float(t[outside[-1]] + dt) if len(outside) else 0.0
    rise_idx = np.nonzero(y >= 0.9 * setpoint)[0]
    return {
        "t": t, "y": y, "u": u,
        "overshoot_pct": float(overshoot),
        "settling_time_s": settling,
        "rise_time_s": float(t[rise_idx[0]]) if len(rise_idx) else float("inf"),
        "final": float(y[-20:].mean()),
    }


def grid_search(kps=np.arange(0.4, 3.01, 0.2), kis=np.arange(2, 20.1, 1.0), max_overshoot=2.0):
    """Fastest settling PI pair with feedforward whose overshoot stays under the cap."""
    best = None
    for kp, ki in itertools.product(kps, kis):
        g = PIDGains(kp=round(float(kp), 2), ki=round(float(ki), 2), feedforward=1.0)
        r = step_response(g)
        if r["overshoot_pct"] > max_overshoot or abs(r["final"] - 1) > 0.01:
            continue
        key = (r["settling_time_s"], r["overshoot_pct"])
        if best is None or key < best[0]:
            best = (key, g, r)
    return best
