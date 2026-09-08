"""Compute the median-distance lengthscale."""

import numpy as np


def median_lengthscale(
    panels,
    max_points=1000,
    random_state=0,
):
    """Return the median pairwise distance between sampled points."""
    if not panels:
        raise ValueError("panels cannot be empty.")
    if max_points < 2:
        raise ValueError("max_points must be at least 2.")

    random_generator = np.random.default_rng(random_state)
    points_per_month = max(
        1,
        int(np.ceil(max_points / len(panels))),
    )
    sampled_points = []

    for panel in panels:
        points = np.asarray(panel["x"], dtype=float)
        number_to_take = min(points_per_month, len(points))
        rows = random_generator.choice(
            len(points),
            size=number_to_take,
            replace=False,
        )
        sampled_points.append(points[rows])

    sampled_points = np.vstack(sampled_points)

    if len(sampled_points) > max_points:
        rows = random_generator.choice(
            len(sampled_points),
            size=max_points,
            replace=False,
        )
        sampled_points = sampled_points[rows]

    squared_norms = np.sum(
        sampled_points**2,
        axis=1,
    )
    squared_distances = (
        squared_norms[:, None]
        + squared_norms[None, :]
        - 2.0 * sampled_points @ sampled_points.T
    )
    squared_distances = np.maximum(
        squared_distances,
        0.0,
    )

    upper_triangle = np.triu_indices(
        len(sampled_points),
        k=1,
    )
    distances = np.sqrt(
        squared_distances[upper_triangle]
    )
    lengthscale = np.median(distances)

    if not np.isfinite(lengthscale) or lengthscale <= 0:
        raise ValueError(
            "The median distance must be positive and finite."
        )

    return float(lengthscale)
