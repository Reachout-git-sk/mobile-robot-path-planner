import math

import numpy as np
import pytest

from robot_sim import maps, trajectory
from robot_sim.models import DiffDrive, Motor
from robot_sim.pid import PID, PIDGains, TUNED, UNTUNED
from robot_sim.pure_pursuit import PurePursuit
from robot_sim.safety import SafetyLimits, SafetyMonitor
from robot_sim.sim import SimConfig, simulate
from robot_sim.tuning import step_response


# ---- trajectory ---------------------------------------------------------------

def l_path():
    leg1 = np.column_stack([np.linspace(0, 5, 101), np.zeros(101)])
    leg2 = np.column_stack([np.full(100, 5.0), np.linspace(0.05, 5, 100)])
    return np.vstack([leg1, leg2])


def test_profile_starts_and_ends_at_rest_and_respects_limits():
    traj = trajectory.build(l_path(), v_max=1.0, a_max=0.5, a_lat_max=0.6)
    assert traj.v[0] == 0 and traj.v[-1] == 0
    assert traj.v.max() <= 1.0 + 1e-9
    acc = trajectory.accelerations(traj)
    assert np.abs(acc).max() <= 0.5 + 1e-6
    assert np.all(np.diff(traj.t) > 0)


def test_profile_slows_for_the_corner():
    traj = trajectory.build(l_path(), v_max=1.0, a_max=0.5, a_lat_max=0.6)
    corner = np.argmin(np.hypot(traj.xy[:, 0] - 5, traj.xy[:, 1]))
    assert traj.v[corner] < 0.5 * traj.v.max()


def test_straight_line_time_matches_trapezoid():
    xy = np.column_stack([np.linspace(0, 10, 1001), np.zeros(1001)])
    traj = trajectory.build(xy, v_max=1.0, a_max=0.5, a_lat_max=0.6)
    # 2 s to accelerate (1 m), 8 m cruise, 2 s to stop: 12 s
    assert traj.duration == pytest.approx(12.0, abs=0.05)


def test_curvature_of_circle():
    a = np.linspace(0, math.pi, 200)
    xy = np.column_stack([2 * np.cos(a), 2 * np.sin(a)])
    k = trajectory.curvature(xy)
    assert np.allclose(np.abs(k[1:-1]), 0.5, atol=1e-3)


# ---- plant models -------------------------------------------------------------

def test_motor_first_order_step_reaches_63_percent_at_tau():
    m = Motor(tau=0.2, gain=1.0, noise_std=0.0)
    dt, t, speeds = 0.001, 0.0, []
    while t < 0.2 - 1e-12:
        speeds.append(m.step(1.0, dt))
        t += dt
    assert speeds[-1] == pytest.approx(1 - math.exp(-1), abs=0.01)


def test_motor_gain_and_saturation():
    m = Motor(tau=0.01, gain=0.9, max_speed=1.5, noise_std=0.0)
    for _ in range(500):
        m.step(5.0, 0.01)
    assert m.speed == pytest.approx(0.9 * 1.5, rel=1e-3)


def test_stuck_motor():
    m = Motor(noise_std=0.0, stuck=True)
    assert m.step(1.0, 0.01) == 0.0


def test_diff_drive_straight_and_spin():
    r = DiffDrive(0.4)
    for _ in range(100):
        r.step(1.0, 1.0, 0.01)
    assert (r.x, r.y) == pytest.approx((1.0, 0.0))
    r2 = DiffDrive(0.4)
    for _ in range(100):
        r2.step(-0.2, 0.2, 0.01)  # w = 1 rad/s
    assert r2.theta == pytest.approx(1.0)
    assert (r2.x, r2.y) == pytest.approx((0.0, 0.0), abs=1e-9)


def test_diff_drive_arc_is_exact():
    r = DiffDrive(0.4)
    # v = 1, w = 1 => circle of radius 1; a quarter turn ends at (1, 1)
    for _ in range(int(round((math.pi / 2) / 0.01))):
        r.step(0.8, 1.2, 0.01)
    assert r.theta == pytest.approx(math.pi / 2, abs=0.01)
    assert (r.x, r.y) == pytest.approx((1.0, 1.0), abs=0.01)


# ---- PID ----------------------------------------------------------------------

def test_pid_removes_steady_state_error_from_weak_motor():
    resp = step_response(TUNED, setpoint=1.0, duration=3.0, noise=0.0)
    assert resp["final"] == pytest.approx(1.0, abs=0.01)


def test_tuning_reduces_overshoot():
    before = step_response(UNTUNED, setpoint=1.0, duration=3.0, noise=0.0)
    after = step_response(TUNED, setpoint=1.0, duration=3.0, noise=0.0)
    assert before["overshoot_pct"] > 5
    assert after["overshoot_pct"] < 1
    assert after["settling_time_s"] < before["settling_time_s"]


def test_pid_output_clamped_and_anti_windup():
    pid = PID(PIDGains(kp=1, ki=10, feedforward=0), out_min=-1, out_max=1)
    for _ in range(1000):
        out = pid.update(10.0, 0.0, 0.01)  # impossible target, output pinned
    assert out == 1.0
    assert pid.integral < 0.5  # did not keep accumulating while saturated


def test_no_derivative_kick_on_setpoint_step():
    pid = PID(PIDGains(kp=0, ki=0, kd=1.0, feedforward=0))
    pid.update(0.0, 0.0, 0.01)
    assert pid.update(5.0, 0.0, 0.01) == 0.0


# ---- pure pursuit & safety ----------------------------------------------------

def test_pure_pursuit_steers_back_to_line():
    xy = np.column_stack([np.linspace(0, 20, 401), np.zeros(401)])
    traj = trajectory.build(xy, 1.0, 0.5, 0.6)
    pp = PurePursuit(traj, wheel_base=0.4)
    robot = DiffDrive(0.4, 0.0, 0.3, 0.0)  # 0.3 m left of the path
    for _ in range(800):
        vl, vr, info = pp.command(*robot.pose)
        robot.step(vl, vr, 0.02)
    assert abs(robot.y) < 0.02


def test_pure_pursuit_turns_right_when_left_of_path():
    xy = np.column_stack([np.linspace(0, 20, 401), np.zeros(401)])
    traj = trajectory.build(xy, 1.0, 0.5, 0.6)
    pp = PurePursuit(traj, wheel_base=0.4)
    pp.idx = 40
    vl, vr, _ = pp.command(2.0, 0.5, 0.0)
    assert vl > vr


def test_lookahead_grows_with_speed():
    traj = trajectory.build(np.column_stack([np.linspace(0, 5, 50), np.zeros(50)]), 1.0, 0.5, 0.6)
    pp = PurePursuit(traj, 0.4, lookahead_min=0.4, lookahead_gain=0.6, lookahead_max=1.5)
    assert pp.lookahead(0.0) == 0.4
    assert pp.lookahead(1.0) == pytest.approx(1.0)
    assert pp.lookahead(5.0) == 1.5


def test_safety_monitor_reasons():
    m = maps.open_room()
    s = SafetyMonitor(m, SafetyLimits(max_tracking_error=0.2, robot_radius=0.25))
    assert s.check(0.0, 2.0, 6.0, 0.05) is None
    assert "tracking error" in s.check(1.0, 2.0, 6.0, 0.3)
    assert s.stopped_at == 1.0
    s2 = SafetyMonitor(m, SafetyLimits(robot_radius=0.25))
    assert "obstacle" in s2.check(0.0, 9.0, 6.0, 0.0)


# ---- whole loop ---------------------------------------------------------------

@pytest.mark.parametrize("name", list(maps.MAPS))
def test_reaches_goal_on_every_map(name):
    res = simulate(maps.load(name))
    assert res.reached_goal, res.stop_reason
    assert res.tracking_error.max() < 0.2
    assert res.stop_reason is None


def test_stuck_wheel_triggers_safety_stop_without_collision():
    m = maps.load("warehouse")
    res = simulate(m, SimConfig(stuck_wheel_at=8.0))
    assert not res.reached_goal
    assert "tracking error" in res.stop_reason
    # after the stop the commanded wheel speeds are zero
    assert res.log["vl_target"][-1] == 0 and res.log["vr_target"][-1] == 0
    for x, y in zip(res.log["x"], res.log["y"]):
        assert not SafetyMonitor(m)._footprint_hits_obstacle(x, y)


def test_simulation_is_reproducible():
    a = simulate(maps.load("open_room"), SimConfig(seed=5)).summary()
    b = simulate(maps.load("open_room"), SimConfig(seed=5)).summary()
    assert a == b
