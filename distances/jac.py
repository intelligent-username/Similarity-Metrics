# Jaccard Distance
# The nice thing about Python is it makes operations like this really easy to implement.

def jaccard_distance(set1, set2) -> float:
    """
    Calculates the jaccard distance between two sets.
    Inputs:
    - set1: The first set to compare.
    - set2: The second set to compare.
    Typically set2 will be the 'authoritative set'

    Returns:
    - float: The Jaccard distance between the two sets.

    Usage:
    >>> set1 = {1, 2, 3}
    >>> set2 = {2, 3, 4}
    >>> jaccard_distance(set1, set2)
    0.5         # The two sets have {2,3} in common and their union is {1,2,3,4}

    >>> set1 = {'yo', 900, 'd', 'e'}
    >>> set2 = {'d', 901, 76.5, 'e'}
    >>> jaccard_distance(set1, set2)
    0.33333         # The two sets have {d, e} in common and their union is {yo, 900, d, e, 901, 76.5}
    """
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    jac = 1 - intersection / union if union > 0 else 0
    return jac
