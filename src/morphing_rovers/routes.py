"""Routes driven by the rover and the switching schedule that drives them the fastest.

A route is the sequence of terrain patches a rover meets on a scenario, sampled every pixel along its path. Given the
masks, it tabulates the velocity of each mode all along the path, so that simulating a scenario only takes a few table
lookups, and so does finding out when the rover should morph: the morph is only ever towards the best mode for the
terrain under the rover, but accepting it locks that mode for the whole cooldown, which is worth it or not depending
on the terrain ahead. `SwitchOracle` answers this by dynamic programming over the whole route.
"""

import logging
from dataclasses import dataclass

import numpy as np
import torch

from morphing_rovers.chromosome import assemble, split
from morphing_rovers.config import ScheduleConfig
from morphing_rovers.experts import Expert
from morphing_rovers.masks import fit_masks_to_schedule, velocity
from morphing_rovers.udp import (
    DELTA_TIME,
    MAX_DA,
    MAX_DV,
    MAX_TIME,
    MAX_VELOCITY,
    MODE_COOLDOWN,
    MODE_VIEW_LEFT,
    MODE_VIEW_RIGHT,
    NUMBER_OF_MODES,
    SAMPLE_RADIUS,
    SCENARIO_POSITIONS,
    SCENARIOS_PER_MAP,
    SIM_TIME_STEPS,
    MysteriousMars,
    Rover,
    morphing_rover_UDP,
    scenario_ids,
)

logger = logging.getLogger(__name__)

# Cells per pixel of the dynamic programming grid: finer than the views, so that the rounding of the rover advance
# at each step does not add up along the route.
GRID = 2
# Cooldown set by a morph. It is decremented during the same step, so the next step starts at MODE_COOLDOWN.
COOLDOWN = MODE_COOLDOWN + 1


@dataclass
class Route:
    """Path of a rover on a scenario, sampled every pixel of arc length until the sample is reached."""

    scenario: int
    positions: np.ndarray  # (K, 2)
    views: torch.Tensor  # mode views seen from these positions (K, 11, 11)
    sample: np.ndarray  # position of the sample
    distance: float  # straight distance from the landing site to the sample
    map_exit: float = float("inf")  # arc length where the rover leaves the map, ending the scenario

    @property
    def arrival(self) -> float:
        """Arc length where the sample gets within reach."""
        return float(len(self.positions) - 1)

    def arc_length(self, position: np.ndarray) -> float:
        """Arc length of the point of the route the closest to `position`."""
        closest = int(np.argmin(((self.positions - position) ** 2).sum(axis=1)))
        following = min(closest + 1, len(self.positions) - 1)
        # Within the pixel, by projecting onto the direction of the route.
        offset = np.dot(position - self.positions[closest], self.positions[following] - self.positions[closest])
        return float(np.clip(closest + offset, 0.0, self.arrival))

    def velocities(self, masks: np.ndarray) -> np.ndarray:
        """Velocity factor (NUMBER_OF_MODES, K) of each mode along the route."""
        with torch.no_grad():
            return velocity(torch.as_tensor(masks, dtype=torch.float32)[:, None], self.views).numpy()


def mode_views(env: MysteriousMars, map_id: int, positions: np.ndarray, headings: np.ndarray) -> torch.Tensor:
    """Mode views (N, 11, 11) of `MysteriousMars.extract_local_view` at N rover poses, in a single batch.

    The challenge rotates a whole view for each pose: only sampling the mode view with the same bilinear interpolation
    is two orders of magnitude faster, which is what makes it possible to search over many routes.
    """
    height_map = env.heightmaps[map_id].numpy()
    columns = np.rint(positions[:, 0]).astype(int)[:, None, None]
    rows = env.heightmap_sizes[map_id][0] - np.rint(positions[:, 1]).astype(int)[:, None, None]
    down, right = np.mgrid[-MODE_VIEW_LEFT:MODE_VIEW_RIGHT, -MODE_VIEW_LEFT:MODE_VIEW_RIGHT]
    # The view is rotated so that the heading of the rover points upwards.
    rotation = np.pi / 2 - headings[:, None, None]
    x = columns + np.cos(rotation) * right - np.sin(rotation) * down
    y = rows + np.sin(rotation) * right + np.cos(rotation) * down
    left, top = np.floor(x).astype(int), np.floor(y).astype(int)
    weight_x, weight_y = x - left, y - top
    views = ((1 - weight_y) * (1 - weight_x) * height_map[top, left] + (1 - weight_y) * weight_x * height_map[top, left + 1]
             + weight_y * (1 - weight_x) * height_map[top + 1, left] + weight_y * weight_x * height_map[top + 1, left + 1])
    return torch.as_tensor(views - views[:, MODE_VIEW_LEFT, MODE_VIEW_LEFT, None, None] + 1, dtype=torch.float32)


def first_views(env: MysteriousMars) -> torch.Tensor:
    """Rover views (30, 89, 89) on the first step of every scenario, which is where they can be told apart."""
    return torch.stack([env.extract_local_view(SCENARIO_POSITIONS[map_id][scenario_id][:2], 0, map_id)[0]
                        for map_id, scenario_id in scenario_ids()])


def build_route(
    env: MysteriousMars, scenario: int, positions: np.ndarray, angles: np.ndarray, left_map: bool = False
) -> Route:
    """Route following the steps of a rover: `positions` (T, 2) and the angle it leaves each of them with (T,).

    The last leg heads straight for the sample, which a rover running out of time would not have reached. With
    `left_map`, the step after the last position took the rover out of the map, which ends the scenario.
    """
    map_id, scenario_id = divmod(scenario, SCENARIOS_PER_MAP)
    start, sample = SCENARIO_POSITIONS[map_id][scenario_id].numpy().astype(np.float64).reshape(2, 2)
    bearing = np.arctan2(*(sample - positions[-1])[::-1])
    points = np.concatenate([positions, [sample]])
    headings = np.concatenate([angles, [bearing]])

    legs = np.diff(points, axis=0)
    lengths = np.maximum(np.linalg.norm(legs, axis=1), 1e-9)
    ends = np.concatenate([[0.0], np.cumsum(lengths)])
    arc = np.arange(0.0, ends[-1], 1.0)
    leg = np.searchsorted(ends, arc, side="right") - 1
    path = points[leg] + ((arc - ends[leg]) / lengths[leg])[:, None] * legs[leg]

    within_reach = np.linalg.norm(path - sample, axis=1) <= SAMPLE_RADIUS
    end = int(np.argmax(within_reach)) + 1 if within_reach.any() else len(path)
    path, leg = path[:end], leg[:end]
    return Route(scenario, path, mode_views(env, map_id, path, headings[leg]), sample,
                 float(np.linalg.norm(sample - start)), float(ends[-2]) if left_map else float("inf"))


class Trajectory:
    """`on_step` hook of `run_single_scenario` recording the positions and angles of a rover."""

    def __init__(self):
        self.positions = []
        self.angles = []

    def __call__(self, rover: Rover, *_) -> None:
        self.positions.append(rover.position.numpy().astype(np.float64))
        self.angles.append(float(rover.angle))

    def route(self, env: MysteriousMars, scenario: int, distance: float) -> Route:
        """Route of the rover, given the `distance` left at the end of the scenario, as returned by the simulator: the
        sample is out of reach while time remains if and only if the rover left the map."""
        left_map = distance > 0 and len(self.positions) < SIM_TIME_STEPS
        return build_route(env, scenario, np.stack(self.positions), np.array(self.angles), left_map)


class _Pilot:
    """Stands in for the network of a rover: steers like the expert, and morphs whenever it can."""

    def __init__(self, rover: Rover, expert: Expert):
        self.rover = rover
        self.expert = expert

    def __call__(self, landscape: torch.Tensor, state: torch.Tensor, latent: torch.Tensor):
        steering = self.expert.steering(self.rover.position.numpy().astype(np.float64), self.rover.angle) / MAX_DA
        return torch.ones(1), torch.tensor([steering]), latent


def trace_routes(udp: morphing_rover_UDP, masks: np.ndarray, experts: dict[int, Expert]) -> dict[int, Route]:
    """Routes driven by a rover with the given masks when it follows the expert of each scenario exactly."""
    rover = Rover(assemble(masks, split(udp.example())[1]))
    routes = {}
    for scenario, expert in experts.items():
        rover.Control = _Pilot(rover, expert)
        trajectory = Trajectory()
        distance, _ = udp.run_single_scenario(rover, *divmod(scenario, SCENARIOS_PER_MAP), on_step=trajectory)
        routes[scenario] = trajectory.route(udp.env, scenario, float(distance))
    return routes


class SwitchOracle:
    """Fewest steps left from every (arc length, mode, cooldown) of a route, by dynamic programming.

    It has foresight of the whole route and the rover has a limited view, hence an oracle: this is the expert imitated
    by the switch output, and it does not bound what a real rover may reach, as it only knows the rover's route.
    """

    def __init__(self, route: Route, velocities: np.ndarray):
        self.route = route
        self.velocities = velocities
        n_cells = int(route.arrival * GRID) + 1
        cells = velocities[:, np.minimum(((np.arange(n_cells) + 0.5) / GRID).astype(int), velocities.shape[1] - 1)]
        best = cells.argmax(axis=0)
        # A rover moves by at least MAX_VELOCITY / 9, more than a cell.
        self.advance = np.rint(MAX_DV * cells * GRID).astype(np.int32)
        self.n_cells = n_cells
        steps = np.zeros((n_cells + 1, NUMBER_OF_MODES, COOLDOWN), dtype=np.int32)
        for cell in range(n_cells - 1, -1, -1):
            for mode in range(NUMBER_OF_MODES):
                following = steps[min(cell + self.advance[mode, cell], n_cells)]
                steps[cell, mode, 1:] = 1 + following[mode, :-1]
                steps[cell, mode, 0] = 1 + min(following[mode, 0], following[best[cell], MODE_COOLDOWN])
        self.steps = steps

    def advantage(self, arc_length: float, mode: int, best: int) -> int:
        """Steps saved by morphing from `mode` to `best` at `arc_length` rather than staying (negative: lost)."""
        cell = min(int(arc_length * GRID), self.n_cells - 1)
        following = min(cell + self.advance[mode, cell], self.n_cells)
        return int(self.steps[following, mode, 0] - self.steps[following, best, MODE_COOLDOWN])


def drive(route: Route, velocities: np.ndarray, oracle: SwitchOracle | None = None) -> tuple[float, np.ndarray]:
    """Simulate a scenario along its route, morphing when the oracle says so, or whenever possible without one.

    Returns its fitness and the mode driven at each view of the route. Views left unvisited, when the time runs out
    or the rover leaves the map, get their fastest mode, for the masks to keep improving there.
    """
    n_views = velocities.shape[1]
    fastest = velocities.argmax(axis=0)
    speed, best = velocities.T.tolist(), fastest.tolist()
    modes = fastest.copy()
    arrival, min_time = route.arrival, route.distance / MAX_VELOCITY
    arc, mode, cooldown = 0.0, 0, 0
    for timestep in range(SIM_TIME_STEPS):
        view = min(int(arc + 0.5), n_views - 1)
        step, used = speed[view][mode], mode
        if cooldown == 0 and best[view] != mode and (oracle is None or oracle.advantage(arc, mode, best[view]) > 0):
            mode = best[view]
            cooldown = COOLDOWN
        if cooldown > 0:
            cooldown -= 1
        arc += MAX_DV * step
        modes[view:min(int(arc + 0.5), n_views - 1) + 1] = used
        if arc >= arrival:
            return timestep * DELTA_TIME / min_time, modes
        if arc >= route.map_exit:
            break
    remaining = np.linalg.norm(route.positions[min(int(arc + 0.5), n_views - 1)] - route.sample) / route.distance
    return float((1 + remaining) * MAX_TIME / min_time), modes


def schedule(masks: np.ndarray, routes: dict[int, Route]) -> tuple[float, dict[int, SwitchOracle], dict[int, np.ndarray]]:
    """Mean fitness of the routes under the oracle's schedule, with the oracles and the modes along each route."""
    oracles, modes, fitness = {}, {}, []
    for scenario, route in routes.items():
        velocities = route.velocities(masks)
        oracles[scenario] = SwitchOracle(route, velocities)
        scenario_fitness, modes[scenario] = drive(route, velocities, oracles[scenario])
        fitness.append(scenario_fitness)
    return float(np.mean(fitness)), oracles, modes


def search_offsets(
    udp: morphing_rover_UDP, masks: np.ndarray, experts: dict[int, Expert], candidates: list[float]
) -> dict[int, float]:
    """Heading offset (degrees) with which each scenario is the fastest, under the oracle's schedule."""
    offsets = {}
    for scenario, expert in experts.items():
        fitness = []
        for offset in candidates:
            candidate = Expert(expert.waypoints, expert.lookahead, offset)
            route = trace_routes(udp, masks, {scenario: candidate})[scenario]
            velocities = route.velocities(masks)
            fitness.append(drive(route, velocities, SwitchOracle(route, velocities))[0])
        offsets[scenario] = candidates[int(np.argmin(fitness))]
        logger.info("scenario %d: heading offset %+.1f, fitness %.4f (straight %.4f)", scenario, offsets[scenario],
                    min(fitness), fitness[candidates.index(0.0)] if 0.0 in candidates else float("nan"))
    return offsets


def refine_masks(masks: np.ndarray, routes: dict[int, Route], cfg: ScheduleConfig) -> np.ndarray:
    """Alternately compute the fastest schedule on the routes and fit the masks to the time spent in each mode.

    Contrary to assigning each view to its fastest mask, the schedule accounts for the cooldown: a mode also drives
    the terrain that follows a morph, whether it suits it or not. Returns the best masks met.
    """
    views = torch.cat([route.views for route in routes.values()])
    # Each scenario counts the same in the fitness, whatever its length.
    weights = torch.cat([torch.full((len(route.views),), 1 / (len(routes) * len(route.views)))
                         for route in routes.values()])
    best_fitness, best_masks = float("inf"), masks
    for round_id in range(cfg.rounds + 1):
        fitness, _, modes = schedule(masks, routes)
        logger.info("mask refinement round %d: fitness %.4f on the routes", round_id, fitness)
        if fitness < best_fitness:
            best_fitness, best_masks = fitness, masks
        if round_id < cfg.rounds:
            assignment = torch.as_tensor(np.concatenate(list(modes.values())))
            masks = fit_masks_to_schedule(masks, views, assignment, weights, cfg.n_iter, cfg.learning_rate)
    return best_masks
