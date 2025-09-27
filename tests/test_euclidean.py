import numpy as np
from distances import euclidean_distance

# 1. Basic 2D known triangle
def test_euclidean_basic_triangle():
    assert euclidean_distance(np.array([0,0]), np.array([3,4])) == 5.0

# 2. 3D example from original combined tests
def test_euclidean_3d_example():
    a = np.array([1,2,3])
    b = np.array([4,6,3])
    assert euclidean_distance(a,b) == 5.0

# 3. Complex numbers example (docstring expectation)
def test_euclidean_complex():
    a = np.array([1+2j, 3+4j])
    b = np.array([5+6j, 7+8j])
    assert euclidean_distance(a,b) == 8.0

# 4. Zero distance (identical points)
def test_euclidean_zero_distance():
    v = np.array([7, -2, 0.5])
    assert euclidean_distance(v,v) == 0.0

# 5. Large magnitude stability (no overflow for typical values)
def test_euclidean_large_values():
    a = np.array([1e6, -1e6])
    b = np.array([1e6 + 3, -1e6 + 4])
    assert euclidean_distance(a,b) == 5.0
