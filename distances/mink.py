# Minkowski distance

import numpy as np

def minkowski_distance(a: np.ndarray, b: np.ndarray, p: float) -> float:
    """
    Calculate the Minkowski distance between two vectors.

    Parameters:
    a (np.ndarray): First vector.
    b (np.ndarray): Second vector.
    p (float): The order of the norm (p >= 1).

    Returns:
    float: The Minkowski distance between vectors a and b.

    Usage:
    >>> minkowski_distance([1, 2], [4, 6], 2)
    5.0 # Same as Euclidean distance
    >>> minkowski_distance([1, 2], [4, 6], 1)
    7.0 # Same as Manhattan distance
    >>> minkowski_distance([1, 2], [4, 6], 3)
    4.497941445275415
    """
    c = (np.abs(a - b)) ** p
    d = np.sum(c)
    e = d ** (1/p)
    return e

if __name__ == "__main__":
    v1, v2 = np.array([1, 2]), np.array([4, 6])
    print(minkowski_distance(v1, v2, 2))  # Expect 5.0
    print(minkowski_distance(v1, v2, 1))  # Expect 7.0
    print(minkowski_distance(v1, v2, 3))  # Expect 4.497941445275415
