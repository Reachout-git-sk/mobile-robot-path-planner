"""Closed-loop simulation: plan -> trajectory -> pure pursuit -> PID -> motors -> robot."""
import math
from dataclasses import dataclass, field

import numpy as np

from . import maps, trajectory
from .astar import plan
from .models import DiffDrive, Motor
from .pid import PID, TUNED, PIDGains
from .pure_pursuit import PurePursuit
from .safety import SafetyLimits, SafetyMonitor
from .smoothing import smooth


@dataclass
class SimConfig:
    dt: float = 0.02
    robot_radius: float = 0.25
    inflation_margin: float = 0.2      # extra clearance on top of the radius
    wheel_base: float = 0.4
    v_max: float = 1.0
    a_max: float = 0.5
    a_lat_max: float = 0.6
    wheel_max: float = 1.5
    motor_tau: float = 0.15
    motor_gain: float = 0.9
    motor_noise: float = 0.01
    pid: PIDGains = field(default_factory=lambda: TUNED)
    lookahead_min: float = 0.4
    lookahead_gain: float = 0.6
    # below the clearance margin, so drifting off the path stops the robot before it can touch anything
    max_tracking_error: float = 0.2
    goal_tolerance: float = 0.15
    seed: int = 0
    stuck_wheel_at: float | None = None   # fault injection: lock the left wheel at this time (s)


@dataclass
class SimResult:
    map: maps.GridMap
    astar_cells: list
    nodes_expanded: int
    traj: trajectory.Trajectory
    log: dict
    reached_goal: bool
    stop_reason: str | None
    sim_time: float

    @property
    def tracking_error(self):
        return np.asarray(self.log["cross_track"])

    def summary(self):
        e = self.tracking_error
        return {
            "map": self.map.name,
            "path_length_m": round(self.traj.length, 2),
            "nodes_expanded": self.nodes_expanded,
            "planned_time_s": round(self.traj.duration, 2),
            "actual_time_s": round(self.sim_time, 2),
            "mean_error_m": round(float(e.mean()), 3),
            "max_error_m": round(float(e.max()), 3),
            "reached_goal": self.reached_goal,
            "stop_reason": self.stop_reason,
        }


def plan_path(gridmap, cfg):
    inflated = gridmap.inflated(cfg.robot_radius + cfg.inflation_margin)
    start = inflated.to_cell(*gridmap.start)
    goal = inflated.to_cell(*gridmap.goal)
    res = plan(inflated.grid, start, goal)
    path = smooth(inflated, res.cells)
    return res, trajectory.build(path, cfg.v_max, cfg.a_max, cfg.a_lat_max)


def simulate(gridmap, cfg=None):
    cfg = cfg or SimConfig()
    rng = np.random.default_rng(cfg.seed)
    res, traj = plan_path(gridmap, cfg)

    x0, y0 = traj.xy[0]
    heading = math.atan2(traj.xy[5, 1] - y0, traj.xy[5, 0] - x0)
    robot = DiffDrive(cfg.wheel_base, x0, y0, heading)
    motors = [Motor(cfg.motor_tau, cfg.motor_gain, cfg.wheel_max, cfg.motor_noise, rng) for _ in range(2)]
    pids = [PID(cfg.pid, -cfg.wheel_max, cfg.wheel_max) for _ in range(2)]
    follower = PurePursuit(traj, cfg.wheel_base, cfg.lookahead_min, cfg.lookahead_gain, wheel_max=cfg.wheel_max)
    safety = SafetyMonitor(gridmap, SafetyLimits(cfg.max_tracking_error, cfg.robot_radius))

    log = {k: [] for k in ("t", "x", "y", "theta", "cross_track", "vl_target", "vr_target",
                           "vl_cmd", "vr_cmd", "vl", "vr")}
    t = 0.0
    timeout = 2 * traj.duration + 10
    reached = False
    gx, gy = traj.xy[-1]

    while t < timeout:
        x, y, th = robot.pose
        if math.hypot(gx - x, gy - y) < cfg.goal_tolerance:
            reached = True
            break

        vl_t, vr_t, info = follower.command(x, y, th)
        if cfg.stuck_wheel_at is not None and t >= cfg.stuck_wheel_at:
            motors[0].stuck = True

        if safety.check(t, x, y, info["cross_track"]):
            vl_t = vr_t = 0.0  # command zero speed and stop the run

        targets = (vl_t, vr_t)
        cmds = [pid.update(tgt, m.speed, cfg.dt) for pid, tgt, m in zip(pids, targets, motors)]
        speeds = [m.step(c, cfg.dt) for m, c in zip(motors, cmds)]
        robot.step(speeds[0], speeds[1], cfg.dt)

        for k, v in (("t", t), ("x", x), ("y", y), ("theta", th), ("cross_track", info["cross_track"]),
                     ("vl_target", vl_t), ("vr_target", vr_t), ("vl_cmd", cmds[0]), ("vr_cmd", cmds[1]),
                     ("vl", speeds[0]), ("vr", speeds[1])):
            log[k].append(v)
        t += cfg.dt
        if safety.stop_reason:
            break

    if not reached and not safety.stop_reason:
        safety.stop_reason = f"timed out after {timeout:.1f} s"
    return SimResult(gridmap, res.cells, res.expanded, traj, {k: np.array(v) for k, v in log.items()},
                     reached, safety.stop_reason, t)
