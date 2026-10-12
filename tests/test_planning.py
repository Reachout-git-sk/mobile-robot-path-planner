import math

import numpy as np
import pytest

from robot_sim import maps
from robot_sim.astar import NoPathError, euclidean, plan, zero
from robot_sim.smoothing import cells_on_segment, chaikin, densify, line_is_free, path_is_free, shortcut, smooth


def open_grid(n=20):
    return np.zeros((n, n), dtype=bool)


def path_cost(cells):
    return sum(math.hypot(a[0] - b[0], a[1] - b[1]) for a, b in zip(cells, cells[1:]))


def test_straight_line_in_open_grid():
    res = plan(open_grid(), (5, 0), (5, 10))
    assert res.cost == pytest.approx(10)
    assert all(r == 5 for r, _ in res.cells)


def test_diagonal_cost():
    res = plan(open_grid(), (0, 0), (7, 7))
    assert res.cost == pytest.approx(7 * math.sqrt(2))


def test_routes_around_wall():
    g = open_grid()
    g[0:15, 10] = True
    res = plan(g, (0, 0), (0, 19))
    assert all(not g[c] for c in res.cells)
    assert max(r for r, _ in res.cells) >= 15


def test_no_corner_cutting():
    g = open_grid(3)
    g[0, 1] = True
    g[1, 0] = True
    with pytest.raises(NoPathError):
        plan(g, (0, 0), (1, 1))


def test_no_path_raises():
    g = open_grid()
    g[:, 10] = True
    with pytest.raises(NoPathError):
        plan(g, (0, 0), (0, 19))


@pytest.mark.parametrize("cell", [(0, 0), (25, 25)])
def test_bad_start(cell):
    g = open_grid()
    g[0, 0] = True
    with pytest.raises(NoPathError):
        plan(g, cell, (5, 5))


@pytest.mark.parametrize("seed", range(15))
def test_astar_matches_dijkstra_on_random_grids(seed):
    """Admissible heuristic => A* cost equals Dijkstra's, with fewer expansions."""
    rng = np.random.default_rng(seed)
    g = rng.random((40, 40)) < 0.25
    g[0, 0] = g[-1, -1] = False
    try:
        ref = plan(g, (0, 0), (39, 39), heuristic=zero)
    except NoPathError:
        with pytest.raises(NoPathError):
            plan(g, (0, 0), (39, 39))
        return
    res = plan(g, (0, 0), (39, 39), heuristic=euclidean)
    assert res.cost == pytest.approx(ref.cost)
    assert res.expanded <= ref.expanded
    assert path_cost(res.cells) == pytest.approx(res.cost)


def test_line_is_free():
    g = open_grid()
    g[5, 5] = True
    assert line_is_free(g, (0, 0), (0, 19))
    assert not line_is_free(g, (0, 0), (10, 10))


def test_segment_traversal_catches_clipped_corner():
    # a shallow line from (2,0) to (0,5) clips the corner of cell (1, 3)
    cells = cells_on_segment((2, 0), (0, 5))
    assert cells[0] == (2, 0) and cells[-1] == (0, 5)
    for a, b in zip(cells, cells[1:]):
        assert abs(a[0] - b[0]) + abs(a[1] - b[1]) <= 2
    g = open_grid()
    for cell in cells[1:-1]:
        g[:] = False
        g[cell] = True
        assert not line_is_free(g, (2, 0), (0, 5)), cell


def test_segment_traversal_straight_and_diagonal():
    assert cells_on_segment((0, 0), (0, 3)) == [(0, 0), (0, 1), (0, 2), (0, 3)]
    diag = cells_on_segment((0, 0), (2, 2))
    assert (1, 1) in diag and (2, 2) == diag[-1]
    assert (0, 1) in diag and (1, 0) in diag  # corner neighbours counted


def test_shortcut_keeps_endpoints_and_reduces_points():
    g = open_grid()
    cells = plan(g, (0, 0), (10, 15)).cells
    pruned = shortcut(g, cells)
    assert pruned[0] == cells[0] and pruned[-1] == cells[-1]
    assert len(pruned) == 2


def test_densify_spacing():
    pts = densify([[0, 0], [1, 0], [1, 1]], 0.1)
    steps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    assert steps.max() <= 0.1 + 1e-9
    assert np.allclose(pts[-1], [1, 1])


def test_chaikin_fixes_endpoints():
    pts = chaikin([[0, 0], [1, 0], [1, 1]], 3)
    assert np.allclose(pts[0], [0, 0]) and np.allclose(pts[-1], [1, 1])
    assert len(pts) > 3


@pytest.mark.parametrize("name", list(maps.MAPS))
def test_every_map_has_a_collision_free_smoothed_path(name):
    m = maps.load(name)
    inflated = m.inflated(0.3)
    res = plan(inflated.grid, inflated.to_cell(*m.start), inflated.to_cell(*m.goal))
    path = smooth(inflated, res.cells)
    assert path_is_free(inflated, path)
    assert np.allclose(path[0], inflated.to_xy(inflated.to_cell(*m.start)))
    smoothed_len = np.linalg.norm(np.diff(path, axis=0), axis=1).sum()
    assert smoothed_len <= res.cost * m.resolution + 1e-6


def test_inflation_grows_obstacles():
    m = maps.open_room()
    assert m.inflated(0.3).grid.sum() > m.grid.sum()


def test_unknown_map():
    with pytest.raises(ValueError, match="unknown map"):
        maps.load("moon")
