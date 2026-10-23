from math import atan2

import numpy as np
import pytest

from morphing_rovers.experts import Expert, arc_points, make_expert
from morphing_rovers.udp import MAX_DA, SCENARIO_POSITIONS

START, END = np.array([100.0, 200.0]), np.array([400.0, 600.0])
LENGTH = 500.0


def deviation_from_chord(points: np.ndarray) -> np.ndarray:
    """Signed distance to the start-end line, positive on the left of the travel direction."""
    chord = (END - START) / LENGTH
    relative = points - START
    return chord[0] * relative[:, 1] - chord[1] * relative[:, 0]


@pytest.mark.parametrize("factor", [0.5, 1.0, 2.0, 5.0])
def test_arc_geometry(factor):
    points = arc_points(START, END, factor, 200)
    radius = factor * LENGTH
    np.testing.assert_allclose(points[[0, -1]], [START, END], atol=1e-9)
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    np.testing.assert_allclose(steps, steps[0])
    # Minor arc bulging to the left, by the sagitta of the circle.
    sagitta = radius - np.sqrt(radius**2 - (LENGTH / 2) ** 2)
    assert deviation_from_chord(points).min() > -1e-9
    assert deviation_from_chord(points).max() == pytest.approx(sagitta, rel=1e-3)


def test_large_radius_is_almost_straight():
    assert deviation_from_chord(arc_points(START, END, 50.0, 200)).max() < 0.01 * LENGTH


def test_radius_too_small():
    with pytest.raises(ValueError, match="at least 0.5"):
        arc_points(START, END, 0.4, 200)


def test_straight_expert_heads_for_the_sample():
    expert = Expert(END, lookahead=5)
    heading = atan2(*(END - START)[::-1])
    assert expert.steering(START, heading - 0.1) == pytest.approx(0.1)
    assert expert.steering(START, heading + 3.0) == -MAX_DA


def test_expert_looks_ahead_along_the_waypoints():
    waypoints = np.stack([np.arange(10.0), np.zeros(10)], axis=1)
    expert = Expert(waypoints, lookahead=3)
    np.testing.assert_array_equal(expert.aim_point(np.array([2.2, 1.0])), [5.0, 0.0])
    np.testing.assert_array_equal(expert.aim_point(np.array([8.0, 1.0])), [9.0, 0.0])


def test_make_expert():
    start, sample = SCENARIO_POSITIONS[1][2].numpy().reshape(2, 2)
    np.testing.assert_array_equal(make_expert(7, None, 200, 5).waypoints, [sample])
    arc = make_expert(7, 2.0, 50, 5).waypoints
    assert arc.shape == (50, 2)
    np.testing.assert_allclose(arc[[0, -1]], [start, sample], atol=1e-3)
