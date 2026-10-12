"""Command line entry point.

    python -m robot_sim run --map warehouse --plot out/warehouse.png --gif out/warehouse.gif
    python -m robot_sim run --map warehouse --stuck-wheel-at 8
    python -m robot_sim benchmark
    python -m robot_sim tune
"""
import argparse
import json
from dataclasses import replace
from pathlib import Path

from . import maps
from .pid import TUNED, UNTUNED
from .sim import SimConfig, simulate
from .tuning import grid_search, step_response


def cmd_run(args):
    cfg = SimConfig(seed=args.seed, stuck_wheel_at=args.stuck_wheel_at, pid=UNTUNED if args.untuned else TUNED)
    res = simulate(maps.load(args.map), cfg)
    print(json.dumps(res.summary(), indent=2))
    if args.plot:
        from .plots import plot_run
        Path(args.plot).parent.mkdir(parents=True, exist_ok=True)
        plot_run(res, args.plot, cfg.max_tracking_error)
        print(f"wrote {args.plot}")
    if args.gif:
        from .plots import animate
        Path(args.gif).parent.mkdir(parents=True, exist_ok=True)
        animate(res, args.gif)
        print(f"wrote {args.gif}")


def cmd_benchmark(args):
    rows = []
    for name in maps.MAPS:
        for label, gains in (("untuned", UNTUNED), ("tuned", TUNED)):
            errs, maxes, reached = [], [], 0
            for seed in range(args.seeds):
                s = simulate(maps.load(name), replace(SimConfig(), seed=seed, pid=gains)).summary()
                errs.append(s["mean_error_m"])
                maxes.append(s["max_error_m"])
                reached += s["reached_goal"]
            rows.append((name, label, sum(errs) / len(errs), max(maxes), f"{reached}/{args.seeds}"))

    print(f"{'map':<11} {'gains':<8} {'mean err (cm)':>13} {'max err (cm)':>12} {'reached':>8}")
    for name, label, mean, mx, reached in rows:
        print(f"{name:<11} {label:<8} {mean * 100:>13.1f} {mx * 100:>12.1f} {reached:>8}")

    before, after = step_response(UNTUNED), step_response(TUNED)
    print(f"\nwheel step response: overshoot {before['overshoot_pct']:.1f}% -> {after['overshoot_pct']:.1f}%, "
          f"settling {before['settling_time_s']:.2f} s -> {after['settling_time_s']:.2f} s")
    if args.plot_step:
        from .plots import plot_step
        Path(args.plot_step).parent.mkdir(parents=True, exist_ok=True)
        plot_step(before, after, args.plot_step)
        print(f"wrote {args.plot_step}")


def cmd_tune(args):
    (settle, overshoot), gains, _ = grid_search()
    print(f"best PI gains with feedforward: kp={gains.kp} ki={gains.ki} "
          f"(settling {settle:.2f} s, overshoot {overshoot:.1f}%)")


def main(argv=None):
    p = argparse.ArgumentParser(prog="robot_sim")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="simulate one map")
    r.add_argument("--map", default="warehouse", choices=list(maps.MAPS))
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--untuned", action="store_true", help="use the pre-tuning PID gains")
    r.add_argument("--stuck-wheel-at", type=float, help="lock the left wheel at this time (s)")
    r.add_argument("--plot")
    r.add_argument("--gif")
    r.set_defaults(func=cmd_run)

    b = sub.add_parser("benchmark", help="tracking error on every map, tuned vs untuned")
    b.add_argument("--seeds", type=int, default=5)
    b.add_argument("--plot-step")
    b.set_defaults(func=cmd_benchmark)

    t = sub.add_parser("tune", help="grid search PI gains on the wheel step response")
    t.set_defaults(func=cmd_tune)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
