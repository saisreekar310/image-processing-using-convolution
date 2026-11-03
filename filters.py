"""Common image-processing kernels and beginner-friendly descriptions."""

import numpy as np


FILTER_KERNELS = {
    "Identity": np.array(
        [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        dtype=float,
    ),
    "Box blur": np.full((3, 3), 1 / 9, dtype=float),
    "Gaussian blur": np.array(
        [[1, 2, 1], [2, 4, 2], [1, 2, 1]],
        dtype=float,
    )
    / 16,
    "Sharpen": np.array(
        [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
        dtype=float,
    ),
    "Sobel X": np.array(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=float,
    ),
    "Sobel Y": np.array(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        dtype=float,
    ),
    "Laplacian": np.array(
        [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
        dtype=float,
    ),
}

FILTER_DESCRIPTIONS = {
    "Identity": "Keeps the center pixel unchanged.",
    "Box blur": "Averages the 3 x 3 neighborhood to soften details.",
    "Gaussian blur": "Uses larger weights near the center for a smooth blur.",
    "Sharpen": "Emphasizes the center pixel relative to its neighbors.",
    "Sobel X": "Highlights changes across columns (vertical edges).",
    "Sobel Y": "Highlights changes across rows (horizontal edges).",
    "Laplacian": "Highlights rapid intensity changes in any direction.",
}