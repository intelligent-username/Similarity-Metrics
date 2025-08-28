# Manhattan Distance

import numpy as np

def manhattan_distance(point1: np.ndarray, point2: np.ndarray) -> float:
    """
    Calculate the Manhattan distance between two n-dimensional points.
    
    Parameters:
        point1 (array-like): Coordinates of the first point.
        point2 (array-like): Coordinates of the second point.
    
    Returns:
        float: Manhattan distance between point1 and point2.

    Usage:
    >>> manhattan_distance([1, 2], [4, 6])
    7.0
    >> manhattan_distance([1, 2, 3], [-4, -5, -6])
    21.0
    """
    # Assume correct inputs
    
    return np.sum( np.abs(point1 - point2) )

if __name__ == "__main__":
    p1, p2 = np.array([1, 2]), np.array([4, 6])
    print(manhattan_distance(p1, p2))  # Expect 7
    p1, p2 = np.array([1, 2, 3]), np.array([-4, -5, -6])
    print(manhattan_distance(p1, p2))  # Expect 21