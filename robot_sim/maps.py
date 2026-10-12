"""Occupancy grid maps. True = occupied. Resolution is metres per cell."""
from dataclasses import dataclass

import numpy as np


@dataclass
class GridMap:
    name: str
    grid: np.ndarray          # bool, shape (rows, cols), row 0 at y = 0
    resolution: float         # m per cell
    start: tuple              # (x, y) in metres
    goal: tuple

    @property
    def shape(self):
        return self.grid.shape

    @property
    def size_m(self):
        rows, cols = self.grid.shape
        return cols * self.resolution, rows * self.resolution

    def to_cell(self, x, y):
        return int(y / self.resolution), int(x / self.resolution)

    def to_xy(self, cell):
        r, c = cell
        return (c + 0.5) * self.resolution, (r + 0.5) * self.resolution

    def in_bounds(self, cell):
        r, c = cell
        rows, cols = self.grid.shape
        return 0 <= r < rows and 0 <= c < cols

    def occupied_xy(self, x, y):
        cell = self.to_cell(x, y)
        return not self.in_bounds(cell) or bool(self.grid[cell])

    def inflated(self, radius_m):
        """Grow obstacles by the robot radius so the planner can treat the robot as a point."""
        k = int(np.ceil(radius_m / self.resolution))
        if k <= 0:
            return self
        out = self.grid.copy()
        rr, cc = np.nonzero(self.grid)
        offsets = [(dr, dc) for dr in range(-k, k + 1) for dc in range(-k, k + 1) if dr * dr + dc * dc <= k * k]
        rows, cols = self.grid.shape
        for dr, dc in offsets:
            r2 = np.clip(rr + dr, 0, rows - 1)
            c2 = np.clip(cc + dc, 0, cols - 1)
            out[r2, c2] = True
        return GridMap(self.name, out, self.resolution, self.start, self.goal)


def _empty(w_m, h_m, res):
    g = np.zeros((int(round(h_m / res)), int(round(w_m / res))), dtype=bool)
    g[0, :] = g[-1, :] = g[:, 0] = g[:, -1] = True  # outer walls
    return g


def _box(g, res, x0, y0, x1, y1):
    g[int(y0 / res):int(np.ceil(y1 / res)), int(x0 / res):int(np.ceil(x1 / res))] = True


def open_room(res=0.1):
    g = _empty(20, 12, res)
    _box(g, res, 9, 4, 11, 8)
    return GridMap("open_room", g, res, (1.5, 6.0), (18.5, 6.0))


def warehouse(res=0.1):
    """Rows of shelving with aisles; the goal is in a different aisle."""
    g = _empty(20, 12, res)
    for x in (4, 8, 12, 16):
        _box(g, res, x, 1.8, x + 1.0, 10.2)
    # cross aisle gaps
    for x in (4, 12):
        _box(g, res, x, 5.4, x + 1.0, 6.6)
        g[int(5.4 / res):int(6.6 / res), int(x / res):int((x + 1.0) / res)] = False
    return GridMap("warehouse", g, res, (1.5, 1.0), (18.5, 11.0))


def maze(res=0.1):
    g = _empty(20, 12, res)
    _box(g, res, 4, 0, 4.4, 8.5)
    _box(g, res, 8, 3.5, 8.4, 12)
    _box(g, res, 12, 0, 12.4, 8.5)
    _box(g, res, 16, 3.5, 16.4, 12)
    _box(g, res, 4, 8.1, 6.6, 8.5)
    _box(g, res, 9.8, 3.5, 12.0, 3.9)
    return GridMap("maze", g, res, (1.5, 1.5), (18.5, 1.5))


def cluttered(res=0.1, seed=3):
    """Random round obstacles, seeded so results are reproducible."""
    rng = np.random.default_rng(seed)
    g = _empty(20, 12, res)
    rows, cols = g.shape
    yy, xx = np.mgrid[0:rows, 0:cols]
    start, goal = (1.5, 6.0), (18.5, 6.0)
    placed = 0
    while placed < 28:
        cx, cy, r = rng.uniform(3, 17), rng.uniform(1, 11), rng.uniform(0.3, 0.8)
        if min(np.hypot(cx - start[0], cy - start[1]), np.hypot(cx - goal[0], cy - goal[1])) < 2.0:
            continue
        g |= (xx * res - cx) ** 2 + (yy * res - cy) ** 2 <= r * r
        placed += 1
    return GridMap("cluttered", g, res, start, goal)


MAPS = {"open_room": open_room, "warehouse": warehouse, "maze": maze, "cluttered": cluttered}


def load(name):
    try:
        return MAPS[name]()
    except KeyError:
        raise ValueError(f"unknown map {name!r}; choose from {', '.join(MAPS)}") from None
