import numpy as np
import math
import pandas as pd


def get_coordinates(scenario_number):
    coordinates = pd.read_csv("../../data/coordinates.txt", sep="\t", header=None)
    coors = coordinates.loc[scenario_number][1:].values
    start, end = np.array([coors[0], coors[1]]), np.array([coors[2], coors[3]])

    return start, end


def get_centres(p, q, rad):
    dist_x, dist_y = q[0] - p[0], q[1] - p[1]
    l = np.sqrt(dist_x ** 2 + dist_y ** 2)

    theta = math.asin(l / (2 * rad))
    h = rad * np.cos(theta)

    c1 = p[0] + dist_x / 2 - h * (dist_y / l)
    c2 = p[1] + dist_y / 2 + h * (dist_x / l)
    c = np.array([c1, c2])

    return c


def get_arc(p, q, c, r, counter_arc=False, num_points=200):
    factor = 0
    if counter_arc:
        factor = 2 * np.pi

    start = 2 * math.atan((p[1] - c[1]) / (p[0] - c[0] + r)) + factor
    end = 2 * math.atan((q[1] - c[1]) / (q[0] - c[0] + r))

    arc_points = np.array([c + r * np.array([np.cos(theta), np.sin(theta)]) for theta in
                           np.arange(start, end, ((end - start) / num_points))])

    return start, end, arc_points


def compute_both_arcs(q, p, radius):
    """compute both  arcs from q to p, given radius. Point q is the start point and p is the end point."""
    c1 = get_centres(p, q, radius)
    c2 = get_centres(q, p, radius)

    start_one, end_one, arc_points_one = get_arc(p, q, c1, radius)
    _, _, arc_points_two = get_arc(p, q, c2, radius)

    if start_one > end_one:
        start_one, end_one, arc_points_one = get_arc(q, p, c1, radius, True)

    return arc_points_one, np.flip(arc_points_two)


def get_closest_arc_point(rover_position, arc):

    dist = np.sqrt(np.sum((rover_position - arc) ** 2, axis=1))
    closest_point = np.argmin(dist)
    #  print(len(arc) - closest_point)
    # if len(arc) - closest_point > 5:
    #     closest_point += 5
    closest_point = arc[closest_point]
    # closest_point = arc[-1]
    return closest_point
