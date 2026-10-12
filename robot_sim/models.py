"""Plant models: wheel motors and the differential-drive body."""
import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Motor:
    """First-order wheel motor: speed lags the command with time constant tau.

    gain < 1 models a motor that delivers less than commanded (load, battery
    sag), which the PID's integral term has to make up. Speeds are wheel rim
    speeds in m/s.
    """
    tau: float = 0.15
    gain: float = 0.9
    max_speed: float = 1.5
    noise_std: float = 0.01
    rng: np.random.Generator = field(default_factory=np.random.default_rng)
    speed: float = 0.0
    stuck: bool = False   # fault injection: wheel locks up

    def step(self, command, dt):
        if self.stuck:
            self.speed = 0.0
            return self.speed
        target = self.gain * float(np.clip(command, -self.max_speed, self.max_speed))
        alpha = 1.0 - math.exp(-dt / self.tau)  # exact discretisation of the lag
        self.speed += alpha * (target - self.speed)
        if self.noise_std:
            self.speed += self.rng.normal(0.0, self.noise_std)
        return self.speed


@dataclass
class DiffDrive:
    """Unicycle kinematics driven by left and right wheel speeds."""
    wheel_base: float = 0.4
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0

    def step(self, v_left, v_right, dt):
        v = 0.5 * (v_left + v_right)
        w = (v_right - v_left) / self.wheel_base
        if abs(w) < 1e-9:
            self.x += v * math.cos(self.theta) * dt
            self.y += v * math.sin(self.theta) * dt
        else:
            # integrate along the arc, exact for constant v and w over the step
            r = v / w
            th1 = self.theta + w * dt
            self.x += r * (math.sin(th1) - math.sin(self.theta))
            self.y -= r * (math.cos(th1) - math.cos(self.theta))
            self.theta = th1
        self.theta = (self.theta + math.pi) % (2 * math.pi) - math.pi
        return v, w

    @property
    def pose(self):
        return self.x, self.y, self.theta
