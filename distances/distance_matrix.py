# Makes Distance Matrices

import numpy as np
from functools import partial

# To avoid annoying import errors in ipynbs

try:
    from .euc import euclidean_distance
    from .manh import manhattan_distance
    from .cos_sim import cosine_similarity
    from .mink import minkowski_distance
    from .gower import gower_distance
    from .jac import jaccard_distance
except ImportError:
    from euc import euclidean_distance
    from manh import manhattan_distance
    from cos_sim import cosine_similarity
    from mink import minkowski_distance
    from gower import gower_distance
    from jac import jaccard_distance

def make_distance_matrix(data, metric='gower', p=3, data_range=None) -> np.ndarray:
    """
    Constructs a distance matrix for a given dataset `data` using the specified distance `metric`. z
    Input:
    - data: A 2D numpy array where each element represents a  point.
    - metric: A string specifying the distance metric to use. Options are 'euclidean', 'manhattan', 'cosine', 'minkowski', and 'gower'. Gower distance is chosen by default. Note that gower distance returns manhattan distance for purely numerical data.
    Output:
    - An nxn 2D numpy array representing the distance matrix.
    """

    metrics = {
        'euclidean': euclidean_distance,
        'manhattan': manhattan_distance,
        'cosine':    lambda a, b: 1.0 - cosine_similarity(a, b),  # distance in [0, 1]
        'jaccard': jaccard_distance
    }

    n = data.shape[0]

    if metric == 'gower':
        def compute_ranges(arr: np.ndarray) -> np.ndarray:
            num_features = arr.shape[1]
            ranges = np.zeros(num_features, dtype=float)
            for i in range(num_features):
                col = arr[:, i]
                # Elementwise numeric detection to support object-dtype columns
                numeric_mask = np.array([np.issubdtype(np.asarray(v).dtype, np.number) for v in col])
                numeric_vals = col[numeric_mask].astype(float)
                ranges[i] = (np.max(numeric_vals) - np.min(numeric_vals)) if numeric_vals.size > 0 else 0.0
            return ranges

        data_range = compute_ranges(data)
        func = partial(gower_distance, data_range=data_range)

    elif metric == 'minkowski':
        func = partial(minkowski_distance, p=p)

    else:
        if metric not in metrics:
            raise ValueError(f"Invalid distance metric: {metric}")
        func = metrics[metric]

    matrix = np.zeros((n, n), dtype=float)

    for x in range(n):
        for y in range(x, n):
            if x == y:
                d = 0.0
            else:
                d = float(func(data[x], data[y]))
            matrix[x, y] = d
            matrix[y, x] = d

    return matrix

if __name__ == "__main__":
    # Test all distances
    p1 = np.array([1, 2, 3])
    p2 = np.array([4, 5, 6])
    print("Point 1 is", p1)
    print("Point 2 is", p2)
    print("Now, the distances")
    print("Euclidean distance is", euclidean_distance(p1, p2))
    print("Manhattan distance is", manhattan_distance(p1, p2))
    print("Cosine similarity is", cosine_similarity(p1, p2))
    print("Minkowski distance (p=3) is", minkowski_distance(p1, p2, p=3))
    print("Gower distance is", gower_distance(p1, p2, [3,3,3]))
    print("Jaccard distance is", jaccard_distance(p1, p2))
    
    print("All distance metrics work.")


