"""Expert steering imitated by the rover controller: head straight for the sample, or follow a circular arc."""

from math import atan2

import numpy as np

from morphing_rovers.udp import MAX_DA, SCENARIO_POSITIONS, SCENARIOS_PER_MAP, minimal_angle_diff


def arc_points(start: np.ndarray, end: np.ndarray, radius_factor: float, num_points: int) -> np.ndarray:
    """Points (num_points, 2) of the minor circular arc from `start` to `end`, in driving order.

    The radius is `radius_factor` times the start-end distance (at least 0.5, a half circle). The arc bulges to
    the left of the start-to-end direction; large factors give an almost straight line.
    """
    if radius_factor < 0.5:
        raise ValueError(f"radius_factor must be at least 0.5 for an arc to join both points, got {radius_factor}.")
    start, end = np.asarray(start, dtype=np.float64), np.asarray(end, dtype=np.float64)
    chord = end - start
    length = np.linalg.norm(chord)
    radius = radius_factor * length
    left = np.array([-chord[1], chord[0]]) / length
    centre = (start + end) / 2 - np.sqrt(max(radius**2 - (length / 2) ** 2, 0.0)) * left
    first_angle = atan2(start[1] - centre[1], start[0] - centre[0])
    # The centre is on the right of the travel direction, so the arc is driven clockwise.
    angles = first_angle - np.linspace(0, 2 * np.arcsin(min(length / (2 * radius), 1.0)), num_points)
    return centre + radius * np.stack([np.cos(angles), np.sin(angles)], axis=1)


class Expert:
    """Steers towards the waypoint `lookahead` points past the waypoint closest to the rover.

    With a single waypoint, the expert simply heads straight for it.
    """

    def __init__(self, waypoints: np.ndarray, lookahead: int):
        self.waypoints = np.atleast_2d(waypoints)
        self.lookahead = lookahead

    def aim_point(self, position: np.ndarray) -> np.ndarray:
        if len(self.waypoints) == 1:
            return self.waypoints[0]
        closest = int(np.argmin(np.linalg.norm(self.waypoints - position, axis=1)))
        return self.waypoints[min(closest + self.lookahead, len(self.waypoints) - 1)]

    def steering(self, position: np.ndarray, angle: float) -> float:
        """Orientation change towards the aim point, limited to what the rover can turn in one step."""
        aim = self.aim_point(position)
        target_angle = atan2(aim[1] - position[1], aim[0] - position[0])
        return float(np.clip(minimal_angle_diff(target_angle, angle), -MAX_DA, MAX_DA))


def make_expert(scenario: int, radius_factor: float | None, num_points: int, lookahead: int) -> Expert:
    """Expert of a scenario (0-29): straight to the sample if `radius_factor` is None, along an arc otherwise."""
    start, sample = SCENARIO_POSITIONS[scenario // SCENARIOS_PER_MAP][scenario % SCENARIOS_PER_MAP].numpy().reshape(2, 2)
    if radius_factor is None:
        return Expert(sample, lookahead)
    return Expert(arc_points(start, sample, radius_factor, num_points), lookahead)
