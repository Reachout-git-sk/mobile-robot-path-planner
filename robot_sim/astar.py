"""A* on an 8-connected occupancy grid.

Moves cost 1 (straight) or sqrt(2) (diagonal). The heuristic is the Euclidean
distance between cells, which never overestimates the true remaining cost on
this grid, so A* returns a shortest path. Diagonal moves that would cut the
corner of an obstacle are not allowed.
"""
import heapq
import math
from dataclasses import dataclass

SQRT2 = math.sqrt(2.0)
NEIGHBOURS = [
    (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
    (-1, -1, SQRT2), (-1, 1, SQRT2), (1, -1, SQRT2), (1, 1, SQRT2),
]


class NoPathError(Exception):
    pass


@dataclass
class PlanResult:
    cells: list        # [(row, col), ...] start to goal
    cost: float        # in cells
    expanded: int      # nodes popped from the open set


def euclidean(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def zero(a, b):
    return 0.0  # turns A* into Dijkstra; used in tests as the reference


def plan(grid, start, goal, heuristic=euclidean):
    rows, cols = grid.shape
    for name, cell in (("start", start), ("goal", goal)):
        r, c = cell
        if not (0 <= r < rows and 0 <= c < cols):
            raise NoPathError(f"{name} {cell} is outside the map")
        if grid[r, c]:
            raise NoPathError(f"{name} {cell} is inside an obstacle")

    g = {start: 0.0}
    parent = {start: None}
    closed = set()
    # (f, tie-break counter, cell); the counter keeps ordering stable for equal f
    open_heap = [(heuristic(start, goal), 0, start)]
    counter = 1
    expanded = 0

    while open_heap:
        _, _, cur = heapq.heappop(open_heap)
        if cur in closed:
            continue  # stale entry, a cheaper route was already expanded
        closed.add(cur)
        expanded += 1
        if cur == goal:
            path = []
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            return PlanResult(path[::-1], g[goal], expanded)

        r, c = cur
        for dr, dc, step in NEIGHBOURS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols) or grid[nr, nc]:
                continue
            if dr and dc and (grid[r + dr, c] or grid[r, c + dc]):
                continue  # no corner cutting
            nxt = (nr, nc)
            cand = g[cur] + step
            if cand < g.get(nxt, math.inf):
                g[nxt] = cand
                parent[nxt] = cur
                heapq.heappush(open_heap, (cand + heuristic(nxt, goal), counter, nxt))
                counter += 1

    raise NoPathError(f"no path from {start} to {goal}")
