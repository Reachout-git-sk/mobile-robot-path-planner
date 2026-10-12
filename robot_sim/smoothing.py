"""Turn the staircase A* path into a shorter, smoother one."""
import numpy as np


def cells_on_segment(a, b):
    """Every cell the segment between the centres of cells a and b passes through.

    Amanatides & Woo grid traversal. Sampling points along the line can step
    over the corner of a cell; walking cell boundaries cannot.
    """
    (r0, c0), (r1, c1) = a, b
    # work in continuous cell coordinates, cell (r, c) spans [r, r+1) x [c, c+1)
    y, x = r0 + 0.5, c0 + 0.5
    dy, dx = r1 - r0, c1 - c0
    r, c = r0, c0
    out = [(r, c)]
    step_r = (dy > 0) - (dy < 0)
    step_c = (dx > 0) - (dx < 0)
    t_delta_r = abs(1.0 / dy) if dy else np.inf
    t_delta_c = abs(1.0 / dx) if dx else np.inf
    t_max_r = ((r + (step_r > 0)) - y) / dy if dy else np.inf
    t_max_c = ((c + (step_c > 0)) - x) / dx if dx else np.inf
    while (r, c) != (r1, c1):
        if abs(t_max_r - t_max_c) < 1e-12:
            # passing exactly through a corner: count both side cells too
            out.append((r + step_r, c))
            out.append((r, c + step_c))
            r, c = r + step_r, c + step_c
            t_max_r += t_delta_r
            t_max_c += t_delta_c
        elif t_max_r < t_max_c:
            r += step_r
            t_max_r += t_delta_r
        else:
            c += step_c
            t_max_c += t_delta_c
        out.append((r, c))
    return out


def line_is_free(grid, a, b):
    """True if the straight segment between cell centres a and b crosses no occupied cell."""
    return not any(grid[cell] for cell in cells_on_segment(a, b))


def shortcut(grid, cells):
    """Greedy line-of-sight pruning: from each kept point, jump to the farthest visible one."""
    if len(cells) <= 2:
        return list(cells)
    out = [cells[0]]
    i = 0
    while i < len(cells) - 1:
        j = len(cells) - 1
        while j > i + 1 and not line_is_free(grid, cells[i], cells[j]):
            j -= 1
        out.append(cells[j])
        i = j
    return out


def densify(points, spacing):
    """Resample a polyline (N x 2 array, metres) at roughly `spacing` metres."""
    pts = np.asarray(points, dtype=float)
    out = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        seg = np.linalg.norm(b - a)
        n = max(1, int(np.ceil(seg / spacing)))
        for k in range(1, n + 1):
            out.append(a + (b - a) * k / n)
    return np.array(out)


def chaikin(points, iterations=3):
    """Corner cutting: round the sharp turns left by shortcutting. Endpoints stay fixed."""
    pts = np.asarray(points, dtype=float)
    for _ in range(iterations):
        if len(pts) < 3:
            break
        q = 0.75 * pts[:-1] + 0.25 * pts[1:]
        r = 0.25 * pts[:-1] + 0.75 * pts[1:]
        mid = np.empty((2 * (len(pts) - 1), 2))
        mid[0::2], mid[1::2] = q, r
        pts = np.vstack([pts[0], mid[1:-1], pts[-1]])
    return pts


def path_is_free(gridmap, points):
    return not any(gridmap.occupied_xy(x, y) for x, y in points)


def smooth(gridmap, cells, spacing=0.05, rounding=3):
    """Shortcut, round corners (only if that stays collision free), then resample."""
    pruned = shortcut(gridmap.grid, cells)
    pts = np.array([gridmap.to_xy(c) for c in pruned])
    for it in range(rounding, -1, -1):
        cand = chaikin(pts, it) if it else pts
        dense = densify(cand, spacing)
        if path_is_free(gridmap, dense):
            return dense
    return densify(pts, spacing)
