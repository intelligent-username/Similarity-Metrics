# Manhattan Distance

import numpy as np

def manhattan_distance(point1: np.arr, point2: np.arr) -> float:
    """
    Calculate the Manhattan distance between two n-dimensional points.
    
    Parameters:
    point1 (array-like): Coordinates of the first point.
    point2 (array-like): Coordinates of the second point.
    
    Returns:
    float: Manhattan distance between point1 and point2.
    """
    # Assume correct inputs
    
    return np.sum( np.abs(point1 - point2) )
