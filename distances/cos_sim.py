# Cosine Similarity Claculator

import numpy as np

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate the Cosine Similarity between two n-dimensional vectors.
    This measures the overlap of orientation, ignoring magnitude. I.e. the cosine of the angle between the lines.
    Please note the floating-point imprecision that will be present.

    Parameters:
        vec1 (array-like): Coordinates of the first vector.
        vec2 (array-like): Coordinates of the second vector.
    
    Returns:
        float: Cosine similarity between vec1 and vec2.

    Usage:
    >>> cosine_similarity([1, 0], [0, 1])
    0.0
    >>> cosine_similarity([1, 1], [2, 2])
    1.0
    >>> cosine_similart([1,1], [-1,-1])
    -1.0
    >>> cosine_similarity([1,1], [1,0])
    0.7071067811865475

    """
    # np.dot finds the dot product of two vectors
    # np.linalg.norm finds the magnitude (length) of a vector
    return (np.dot(vec1, vec2)
            / ( np.linalg.norm(vec1) * np.linalg.norm(vec2) )
            )

if __name__ == "__main__":
    
    v1, v2 = np.array([1, 0]), np.array([0, 1])
    print(cosine_similarity(v1, v2))  # Expect 0.0
    v1, v2 = np.array([1, 1]), np.array([2, 2])
    print(cosine_similarity(v1, v2))  # Expect 1.0
    v1, v2 = np.array([1, 1]), np.array([-1, -1])
    print(cosine_similarity(v1, v2))  # Expect -1.0
    v1, v2 = np.array([1, 1]), np.array([1, 0])
    print(cosine_similarity(v1, v2))  # Expect ~0.707
