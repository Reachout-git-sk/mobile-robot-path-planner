"""Pure pursuit path follower producing left/right wheel speed targets."""
import math

import numpy as np


class PurePursuit:
    def __init__(self, traj, wheel_base, lookahead_min=0.4, lookahead_gain=0.6,
                 lookahead_max=1.5, v_min=0.08, wheel_max=1.5, search_window=200):
        self.traj = traj
        self.xy = traj.xy
        self.wheel_base = wheel_base
        self.ld_min, self.ld_gain, self.ld_max = lookahead_min, lookahead_gain, lookahead_max
        self.v_min = v_min
        self.wheel_max = wheel_max
        self.window = search_window
        self.idx = 0  # progress along the path; only moves forward

    def nearest(self, x, y):
        """Nearest path point in a forward window, so the robot cannot 'skip back'."""
        hi = min(len(self.xy), self.idx + self.window)
        d = np.hypot(self.xy[self.idx:hi, 0] - x, self.xy[self.idx:hi, 1] - y)
        k = int(np.argmin(d))
        self.idx += k
        return self.idx, float(d[k])

    def lookahead(self, v):
        """Lookahead grows with speed: short when slow for accuracy, long when fast for stability."""
        return float(np.clip(self.ld_min + self.ld_gain * v, self.ld_min, self.ld_max))

    def target_point(self, x, y, ld):
        for j in range(self.idx, len(self.xy)):
            if math.hypot(self.xy[j, 0] - x, self.xy[j, 1] - y) >= ld:
                return j, self.xy[j]
        return len(self.xy) - 1, self.xy[-1]

    def command(self, x, y, theta):
        """Return (v_left, v_right, info) wheel speed targets for the current pose."""
        i, cross_track = self.nearest(x, y)
        v = max(self.v_min, self.traj.v[i])
        ld = self.lookahead(v)
        j, (tx, ty) = self.target_point(x, y, ld)

        alpha = math.atan2(ty - y, tx - x) - theta
        alpha = (alpha + math.pi) % (2 * math.pi) - math.pi
        dist = max(math.hypot(tx - x, ty - y), 1e-6)
        kappa = 2.0 * math.sin(alpha) / dist
        # slow down when pointing away from the target (e.g. after a disturbance)
        v *= max(0.25, math.cos(alpha))

        w = v * kappa
        half = 0.5 * self.wheel_base * w
        vl, vr = v - half, v + half
        # keep the requested curvature if a wheel would saturate
        scale = max(1.0, abs(vl) / self.wheel_max, abs(vr) / self.wheel_max)
        vl, vr = vl / scale, vr / scale
        return vl, vr, {"cross_track": cross_track, "lookahead": ld, "target": (tx, ty), "index": i}
