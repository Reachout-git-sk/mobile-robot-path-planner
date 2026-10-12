"""PID speed controller for one wheel."""
from dataclasses import dataclass


@dataclass
class PIDGains:
    kp: float
    ki: float
    kd: float = 0.0
    feedforward: float = 1.0   # command += ff * setpoint; PID only corrects the error


class PID:
    def __init__(self, gains, out_min=-2.0, out_max=2.0):
        self.g = gains
        self.out_min, self.out_max = out_min, out_max
        self.integral = 0.0
        self.prev_measured = None

    def reset(self):
        self.integral = 0.0
        self.prev_measured = None

    def update(self, setpoint, measured, dt):
        g = self.g
        error = setpoint - measured
        # derivative on the measurement, not the error, so a setpoint step
        # does not produce a derivative kick
        d = 0.0 if self.prev_measured is None else -(measured - self.prev_measured) / dt
        self.prev_measured = measured

        unsat = g.feedforward * setpoint + g.kp * error + g.ki * (self.integral + error * dt) + g.kd * d
        out = min(self.out_max, max(self.out_min, unsat))
        # anti-windup: only integrate when the output is not pinned at a limit
        # in the direction the error would push it further
        if out == unsat or (out == self.out_max and error < 0) or (out == self.out_min and error > 0):
            self.integral += error * dt
        return out


# Hand-picked aggressive gains used as the "before tuning" baseline, and the
# result of tools/tune_pid.py (grid search on the motor step response).
UNTUNED = PIDGains(kp=0.3, ki=30.0, kd=0.0, feedforward=0.0)
TUNED = PIDGains(kp=2.8, ki=16.0, kd=0.0, feedforward=1.0)
