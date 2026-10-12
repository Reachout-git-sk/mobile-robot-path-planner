"""Safety supervisor: decides when the robot must stop, and records why."""
import math
from dataclasses import dataclass


@dataclass
class SafetyLimits:
    max_tracking_error: float = 0.5   # m from the planned path
    robot_radius: float = 0.25        # used for the collision check against the real map


class SafetyMonitor:
    def __init__(self, gridmap, limits=None):
        self.map = gridmap
        self.limits = limits or SafetyLimits()
        self.stop_reason = None
        self.stopped_at = None

    def _footprint_hits_obstacle(self, x, y):
        r = self.limits.robot_radius
        pts = [(x, y)] + [(x + r * math.cos(a), y + r * math.sin(a)) for a in [k * math.pi / 8 for k in range(16)]]
        return any(self.map.occupied_xy(px, py) for px, py in pts)

    def check(self, t, x, y, cross_track):
        if self.stop_reason:
            return self.stop_reason
        if cross_track > self.limits.max_tracking_error:
            self.stop_reason = (f"tracking error {cross_track:.3f} m exceeded limit "
                                f"{self.limits.max_tracking_error:.2f} m")
        elif self._footprint_hits_obstacle(x, y):
            self.stop_reason = "robot footprint touched an obstacle"
        if self.stop_reason:
            self.stopped_at = t
        return self.stop_reason
