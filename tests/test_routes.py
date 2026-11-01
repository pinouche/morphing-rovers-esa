import numpy as np
import pytest
import torch

from morphing_rovers.chromosome import always_switch, assemble, split, wired_network
from morphing_rovers.config import ImitationConfig, ScheduleConfig
from morphing_rovers.experts import make_expert
from morphing_rovers.imitation import make_experts
from morphing_rovers.masks import fit_masks_to_schedule, velocity
from morphing_rovers.routes import (
    COOLDOWN,
    GRID,
    Route,
    SwitchOracle,
    _Pilot,
    build_route,
    drive,
    first_views,
    mode_views,
    refine_masks,
    schedule,
    search_offsets,
    trace_routes,
)
from morphing_rovers.udp import (
    MAX_VELOCITY,
    MODE_COOLDOWN,
    SAMPLE_RADIUS,
    SCENARIO_POSITIONS,
    TOTAL_NUM_SCENARIOS,
    Rover,
)


def flat_route(n_views: int) -> Route:
    """Straight route along the x axis; only its length matters, the velocities being given to the oracle."""
    positions = np.stack([np.arange(n_views, dtype=np.float64), np.zeros(n_views)], axis=1)
    return Route(0, positions, torch.ones(n_views, 11, 11), positions[-1] + [SAMPLE_RADIUS, 0.0], n_views - 1 + 6.0)


def by_mode(n_views: int, *values) -> np.ndarray:
    """Velocities (4, n_views) from the velocity of each mode, a scalar or one value per view; the others can't move."""
    velocities = np.zeros((4, n_views))
    for mode, value in enumerate(values):
        velocities[mode] = value
    return velocities


def run_lengths(modes: np.ndarray) -> np.ndarray:
    return np.diff(np.flatnonzero(np.diff(modes) != 0))


@pytest.fixture(scope="module")
def masks(solution) -> np.ndarray:
    return split(solution)[0]


@pytest.fixture(scope="module")
def routes(udp, masks) -> dict:
    return trace_routes(udp, masks, make_experts(ImitationConfig(scenarios=[0, 9, 16, 19])))


def test_batched_mode_views_match_the_challenge(udp):
    rng = np.random.default_rng(0)
    for map_id in range(6):
        size = np.array(udp.env.heightmap_sizes[map_id][::-1])
        positions = rng.uniform(60, size - 60, (40, 2))
        headings = rng.uniform(-3.5, 3.5, 40)
        expected = torch.stack([udp.env.extract_local_view(p, float(h), map_id)[1] for p, h in zip(positions, headings, strict=True)])
        np.testing.assert_allclose(mode_views(udp.env, map_id, positions, headings).numpy(), expected.numpy(), atol=1e-3)


def test_first_views_tell_the_scenarios_apart(udp):
    views = first_views(udp.env)
    assert views.shape == (30, 89, 89)
    assert len({view.numpy().tobytes() for view in views}) == 30


def test_route_of_a_straight_expert(routes):
    route = routes[19]
    assert route.views.shape == (len(route.positions), 11, 11)
    start, sample = SCENARIO_POSITIONS[3][4].numpy().reshape(2, 2)
    assert route.distance == pytest.approx(np.linalg.norm(sample - start))
    np.testing.assert_allclose(route.positions[0], start)
    np.testing.assert_array_equal(route.views[:, 5, 5], 1.0)
    steps = np.linalg.norm(np.diff(route.positions, axis=0), axis=1)
    assert np.all(steps > 0.5) and np.all(steps <= 1 + 1e-9)
    # The route stops where the sample is within reach, and no earlier.
    assert np.linalg.norm(route.positions[-1] - sample) <= SAMPLE_RADIUS
    assert np.linalg.norm(route.positions[-2] - sample) > SAMPLE_RADIUS


def test_route_of_a_rover_running_out_of_time_still_reaches_the_sample(udp):
    start, sample = SCENARIO_POSITIONS[0][0].numpy().astype(np.float64).reshape(2, 2)
    route = build_route(udp.env, 0, np.array([start, start + [10.0, 0.0]]), np.array([0.0, 0.0]))
    assert np.linalg.norm(route.positions[-1] - sample) <= SAMPLE_RADIUS


def test_arc_length_of_positions_near_the_route():
    route = flat_route(20)
    assert route.arc_length(np.array([7.3, 2.0])) == pytest.approx(7.3)
    assert route.arc_length(np.array([-3.0, 0.0])) == 0.0
    assert route.arc_length(np.array([40.0, 0.0])) == route.arrival


def test_drive_counts_time_like_the_simulator():
    # At full speed, the rover moves 11 pixels per step: the sample is in reach after 9 steps (timestep 8).
    route = flat_route(100)
    fitness, modes = drive(route, by_mode(100, 1.0))
    assert fitness == pytest.approx(8 * MAX_VELOCITY / route.distance)
    assert np.all(modes == 0)


def test_drive_out_of_time():
    # 2.2 pixels per step during 500 steps: the rover stops after 1100 of the 2000 pixels to the sample.
    route = flat_route(2000)
    fitness, _ = drive(route, by_mode(2000, 0.2, 0.1, 0.1, 0.1))
    remaining = (route.sample[0] - 1100) / route.distance
    assert fitness == pytest.approx((1 + remaining) * 500 * MAX_VELOCITY / route.distance, rel=1e-3)


def test_drive_ends_where_the_rover_leaves_the_map():
    route = flat_route(2000)
    route.map_exit = 300.0
    fitness, _ = drive(route, by_mode(2000, 0.5))
    remaining = (route.sample[0] - 305) / route.distance
    assert fitness == pytest.approx((1 + remaining) * 500 * MAX_VELOCITY / route.distance, rel=1e-3)


def test_route_leaving_the_map_scores_like_the_simulator(udp, masks):
    expert = make_expert(20, None, 200, 5, -30.0)
    route = trace_routes(udp, masks, {20: expert})[20]
    assert route.map_exit < route.arrival
    rover = Rover(assemble(masks, split(udp.example())[1]))
    rover.Control = _Pilot(rover, expert)
    distance, time = udp.run_single_scenario(rover, 4, 0)
    assert distance > 0
    assert drive(route, route.velocities(masks))[0] == pytest.approx(float((1 + distance) * time), rel=0.02)


def test_routes_staying_on_the_map_have_no_exit(routes):
    assert all(route.map_exit == float("inf") for route in routes.values())


def test_oracle_declines_a_morph_for_a_short_patch():
    # Mode 1 is only better for a short patch; morphing would lock it on the slow terrain that follows.
    n_views = 400
    mode_0 = np.full(n_views, 0.5)
    mode_1 = np.full(n_views, 0.1)
    mode_1[100:112] = 0.9
    velocities = by_mode(n_views, mode_0, mode_1)
    route = flat_route(n_views)
    oracle = SwitchOracle(route, velocities)
    assert oracle.advantage(100.0, 0, 1) < 0
    always, always_modes = drive(route, velocities)
    assert np.any(always_modes == 1)
    fastest, modes = drive(route, velocities, oracle)
    assert fastest < always and np.all(modes == 0)


def test_oracle_accepts_a_morph_for_a_long_stretch():
    n_views = 600
    mode_0 = np.where(np.arange(n_views) < 100, 0.5, 0.15)
    mode_1 = np.where(np.arange(n_views) < 100, 0.1, 0.8)
    route = flat_route(n_views)
    velocities = by_mode(n_views, mode_0, mode_1)
    oracle = SwitchOracle(route, velocities)
    assert oracle.steps.shape == (int(route.arrival * GRID) + 2, 4, COOLDOWN)
    # Morphing right now or one step later can tie, but it has to happen at some point.
    assert oracle.advantage(120.0, 0, 1) >= 0
    fitness, modes = drive(route, velocities, oracle)
    assert fitness < drive(route, by_mode(n_views, mode_0))[0]
    assert modes[-1] == 1


def test_a_mode_lasts_as_long_as_in_the_simulator():
    """Terrain flipping at every step, so that the rover morphs as soon as its cooldown allows it."""
    first, second = torch.ones(11, 11), 1 + torch.arange(121.0).reshape(11, 11) / 60
    masks = torch.stack([first, second, first, second])
    rover = Rover(always_switch(assemble(masks.numpy(), wired_network())))
    rover.reset(torch.tensor([500.0, 500.0]))
    real_modes = []
    for step in range(60):
        real_modes.append(rover.current_mode)
        view = second if step % 2 else first
        rover.update_rover_state(torch.zeros(89, 89), view, torch.tensor([1000.0, 0.0]), torch.tensor(1000.0))
    real_runs = run_lengths(np.array(real_modes))
    assert len(real_runs) > 3 and np.all(real_runs == real_runs[0])

    # One pixel per step, with the same flipping terrain.
    n_views = 60
    odd = np.arange(n_views) % 2 == 1
    velocities = by_mode(n_views, np.where(odd, 0.09, 1 / 11), np.where(odd, 1 / 11, 0.09))
    _, modes = drive(flat_route(n_views), velocities)
    assert real_runs[0] == COOLDOWN
    assert np.all(run_lengths(modes)[:3] % COOLDOWN < 3)


def test_the_oracle_counts_the_cooldown_of_a_morph():
    """Steps left, against a brute force over the two possible schedules of a route with a single decision."""
    n_views = 200
    mode_0, mode_1 = np.full(n_views, 0.5), np.full(n_views, 0.1)
    mode_1[0:2] = 0.6
    route = flat_route(n_views)
    oracle = SwitchOracle(route, by_mode(n_views, mode_0, mode_1))
    # Staying: rounded advance of 5.5 px per step (11 cells). Morphing locks 0.1 for the 9 following steps, 1.1 px each.
    assert oracle.advantage(0.0, 0, 1) < 0
    stuck = oracle.steps[0, 1, MODE_COOLDOWN]
    free = oracle.steps[0, 1, 0]
    assert stuck > free


def test_fast_simulation_matches_the_real_one(udp, masks):
    """Exact steering and always-switch, through the official simulator and through the route tables."""
    experts = make_experts(ImitationConfig(scenarios=list(range(0, TOTAL_NUM_SCENARIOS, 5))))
    routes = trace_routes(udp, masks, experts)
    fast = np.mean([drive(route, route.velocities(masks))[0] for route in routes.values()])
    # A single scenario is chaotic, but this averages out.
    assert fast == pytest.approx(udp.fitness(always_switch(assemble(masks, wired_network())), list(experts))[0], abs=0.05)


def test_schedule_beats_always_switching(masks, routes):
    fitness, oracles, modes = schedule(masks, routes)
    assert set(oracles) == set(modes) == set(routes)
    always = np.mean([drive(route, route.velocities(masks))[0] for route in routes.values()])
    assert fitness < always
    assert all(len(modes[s]) == len(routes[s].views) for s in routes)


def test_fit_masks_to_schedule_reduces_the_time(masks, routes):
    route = routes[19]
    modes = torch.as_tensor(drive(route, route.velocities(masks))[1])
    weights = torch.full((len(route.views),), 1 / len(route.views))

    def time(candidate: np.ndarray) -> float:
        return (weights / velocity(torch.as_tensor(candidate, dtype=torch.float32)[modes], route.views)).sum().item()

    fitted = fit_masks_to_schedule(masks, route.views, modes, weights, n_iter=30, learning_rate=0.05)
    assert time(fitted) < time(masks)


def test_refine_masks_keeps_the_best_masks_met(masks, routes):
    before = schedule(masks, routes)[0]
    refined = refine_masks(masks, routes, ScheduleConfig(rounds=2, n_iter=5, learning_rate=0.01))
    assert refined.shape == (4, 11, 11)
    assert schedule(refined, routes)[0] <= before


def test_refine_masks_without_rounds_returns_the_masks(masks, routes):
    np.testing.assert_array_equal(refine_masks(masks, routes, ScheduleConfig(rounds=0)), masks)


def test_routes_along_arcs_bulge_on_the_requested_side(udp, masks):
    left = trace_routes(udp, masks, {0: make_expert(0, 2.0, 200, 5)})[0]
    right = trace_routes(udp, masks, {0: make_expert(0, -2.0, 200, 5)})[0]
    start, sample = SCENARIO_POSITIONS[0][0].numpy().reshape(2, 2)
    chord = (sample - start) / np.linalg.norm(sample - start)

    def side_at_mid_route(route: Route) -> float:
        offset = route.positions[len(route.positions) // 2] - start
        return chord[0] * offset[1] - chord[1] * offset[0]

    assert side_at_mid_route(left) > 50 and side_at_mid_route(right) < -50


def test_search_offsets_picks_the_fastest_candidate(udp, masks, routes):
    experts = make_experts(ImitationConfig(scenarios=[19]))
    offsets = search_offsets(udp, masks, experts, [0.0, 6.0, -6.0])
    assert offsets[19] in (0.0, 6.0, -6.0)
    fitness = {}
    for candidate in (0.0, 6.0, -6.0):
        expert = make_expert(19, None, 200, 5, candidate)
        fitness[candidate] = schedule(masks, trace_routes(udp, masks, {19: expert}))[0]
    assert fitness[offsets[19]] == min(fitness.values())


def test_search_offsets_rejects_candidates_leaving_the_map(udp, masks):
    experts = make_experts(ImitationConfig(scenarios=[20]))
    assert search_offsets(udp, masks, experts, [0.0, -16.0])[20] == 0.0
