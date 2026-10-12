import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

PLANNED = "#2563eb"
ACTUAL = "#ea580c"
LEFT = "#0891b2"
RIGHT = "#9333ea"


def draw_map(ax, gridmap):
    w, h = gridmap.size_m
    ax.imshow(gridmap.grid, origin="lower", extent=(0, w, 0, h), cmap="Greys", vmin=0, vmax=1.4, interpolation="nearest")
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")


def plot_run(result, path, max_error=None):
    log, traj = result.log, result.traj
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)

    ax = axes[0, 0]
    draw_map(ax, result.map)
    ax.plot(traj.xy[:, 0], traj.xy[:, 1], color=PLANNED, lw=1.5, label="planned")
    ax.plot(log["x"], log["y"], color=ACTUAL, lw=1.5, ls="--", label="actual")
    ax.plot(*result.map.start, "o", color="#16a34a", ms=8, label="start")
    ax.plot(*result.map.goal, "*", color="#dc2626", ms=12, label="goal")
    ax.set_title(f"{result.map.name}: planned vs actual path")
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[0, 1]
    ax.plot(log["t"], log["cross_track"] * 100, color=ACTUAL)
    if max_error:
        ax.axhline(max_error * 100, color="#dc2626", ls=":", label="safety stop limit")
        ax.legend(fontsize=8)
    s = result.summary()
    ax.set_title(f"Tracking error (mean {s['mean_error_m'] * 100:.1f} cm, max {s['max_error_m'] * 100:.1f} cm)")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("cm")
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(log["t"], log["vl_target"], color=LEFT, lw=1, label="left target")
    ax.plot(log["t"], log["vl"], color=LEFT, lw=1, alpha=0.5, ls="--", label="left actual")
    ax.plot(log["t"], log["vr_target"], color=RIGHT, lw=1, label="right target")
    ax.plot(log["t"], log["vr"], color=RIGHT, lw=1, alpha=0.5, ls="--", label="right actual")
    ax.set_title("Wheel speed: command vs actual")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("m/s")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    v_actual = 0.5 * (log["vl"] + log["vr"])
    ax.plot(traj.t, traj.v, color=PLANNED, label="planned profile")
    ax.plot(log["t"], v_actual, color=ACTUAL, alpha=0.8, label="actual")
    ax.set_title("Forward speed")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("m/s")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    if result.stop_reason:
        fig.suptitle(f"SAFETY STOP: {result.stop_reason}", color="#dc2626")
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_step(before, after, path, labels=("before tuning", "after tuning")):
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for r, label, color in ((before, labels[0], "#9ca3af"), (after, labels[1], PLANNED)):
        ax.plot(r["t"], r["y"], color=color,
                label=f"{label}: {r['overshoot_pct']:.1f}% overshoot, settles in {r['settling_time_s']:.2f} s")
    ax.axhline(1.0, color="k", lw=0.8, ls=":")
    ax.set_xlim(0, 1.5)
    ax.set_title("Wheel speed step response (1 m/s command, motor gain 0.9, tau 0.15 s)")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("wheel speed (m/s)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def animate(result, path, frames=120, fps=15):
    log, traj = result.log, result.traj
    n = len(log["t"])
    idx = np.linspace(0, n - 1, min(frames, n)).astype(int)
    fig, ax = plt.subplots(figsize=(6.4, 4.0), constrained_layout=True)
    draw_map(ax, result.map)
    ax.plot(traj.xy[:, 0], traj.xy[:, 1], color=PLANNED, lw=1, alpha=0.6)
    ax.plot(*result.map.goal, "*", color="#dc2626", ms=10)
    trail, = ax.plot([], [], color=ACTUAL, lw=1.5)
    body = Circle((0, 0), 0.25, color=ACTUAL, zorder=5)
    ax.add_patch(body)
    heading, = ax.plot([], [], color="white", lw=2, zorder=6)
    title = ax.set_title(result.map.name)

    def update(k):
        i = idx[k]
        x, y, th = log["x"][i], log["y"][i], log["theta"][i]
        trail.set_data(log["x"][: i + 1], log["y"][: i + 1])
        body.center = (x, y)
        heading.set_data([x, x + 0.25 * np.cos(th)], [y, y + 0.25 * np.sin(th)])
        title.set_text(f"{result.map.name}  t={log['t'][i]:.1f}s  error={log['cross_track'][i] * 100:.1f} cm")
        return trail, body, heading, title

    anim = FuncAnimation(fig, update, frames=len(idx), blit=False)
    anim.save(path, writer=PillowWriter(fps=fps), dpi=70)
    plt.close(fig)
