# Gower Distance

import numpy as np

def gower_distance(a: np.ndarray, b: np.ndarray, data_range) -> float:
    """
    Calculates the Gower distance between two points.
    Assuming 'nice' inputs (i.e. corresponding elements of a and b are of the same data type, & a and b have the same shape).
    Could add error checking but no point.

    Inputs:
    a: A numpy ararray representing the first point.
    b: A numpy array representing the second point.
    
    Output:
    - A float representing the Gower distance between points a and b.

    Usage:
    >>> a = np.array([1, 2, 3])
    >>> b = np.array([4, 5, 6])
    >>> ranges = np.array([3, 3, 3])
    >>> distance = gower_distance(a, b, ranges)
    >>> distance
    1.0

    >>> a = np.array([1, 'cat', 3])
    >>> b = np.array([4, 'dog', 6])
    >>> ranges = np.array([3, 0, 3])  # Range is 0 for categorical features
    >>> distance = gower_distance(a, b, ranges)
    1.0
    
    >>> a = np.array(['a', 'b','c'])
    >>> b = np.array(['a', 'b','c'])
    >>> ranges = np.array([0, 0, 0])
    >>> distance = gower_distance(a, b, ranges)


    """
    def feature_dist(c, d, i, data_range) -> float:
        """
        Calculates the distance between two specific features c and d.
        """
        if isinstance(c, (int, float)) and isinstance(d, (int, float)):
            # Numerical feature
            if data_range[i] == 0:
                return 0
            else:
                return np.abs(c - d) / data_range[i]
        else:
            # Categorical feature
            return 0 if c == d else 1

    
    distance = 0
    weight = 0

    dim = a.shape[0]
    for i in range(dim):
        distance += feature_dist(a[i], b[i], i, data_range)
        weight += 1 if a[i] is not None and b[i] is not None else 0

    return distance/weight

if __name__ == "__main__":
    p1 = np.array([1, 2, 3])
    p2 = np.array([4, 5, 6])
    data_range = np.array([3, 3, 3])  # max - min for each feature
    print(gower_distance(p1, p2, data_range))  # Expect 1.0

    p1 = np.array([1, 'cat', 3])
    p2 = np.array([4, 'dog', 6])
    data_range = np.array([3, 0, 3])
    print(gower_distance(p1, p2, data_range))  # Expect 1.0

    p1 = np.array(['a', 'b','c'])
    p2 = np.array(['a', 'b','c'])
    data_range = np.array([0, 0, 0])           # Data is identical
    print(gower_distance(p1, p2, data_range))  # Expect 0.0

