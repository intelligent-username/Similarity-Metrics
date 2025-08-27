# Euclidean distance

import numpy as np

def euclidean_distance(point1: np.ndarray, point2: np.ndarray) -> float:
    """
    Calculate the Euclidean distance between two n-dimensional points.
    Accepts complex numbers.
    
    Parameters:
    point1 (array-like): Coordinates of the first point.
    point2 (array-like): Coordinates of the second point.
    
    Returns:
    float: Euclidean distance between point1 and point2.

    Usage: 
    >>> euclidean_distance([1, 2], [4, 6])
    5.0
    >>> euclidean_distance([1+2j, 3+4j], [5+6j, 7+8j])
    8.0
    """
    # Assume correct inputs

    return np.linalg.norm(point1 - point2)
