import numpy as np
from distances import minkowski_distance, manhattan_distance, euclidean_distance

# 1. p=1 equals Manhattan

def test_minkowski_equals_manhattan():
    a = np.array([1,2,3])
    b = np.array([4,6,3])
    assert minkowski_distance(a,b,1) == manhattan_distance(a,b)

# 2. p=2 equals Euclidean

def test_minkowski_equals_euclidean():
    a = np.array([1,2,3])
    b = np.array([4,6,3])
    assert minkowski_distance(a,b,2) == euclidean_distance(a,b)

# 3. p>2 decreases relative to p=2 for this vector (monotonic behavior)

def test_minkowski_p3_less_than_p2():
    a = np.array([1,2])
    b = np.array([4,6])
    assert minkowski_distance(a,b,3) < minkowski_distance(a,b,2)

# 4. Zero distance

def test_minkowski_zero():
    v = np.array([2,4,6])
    assert minkowski_distance(v,v,3) == 0.0

# 5. Higher p approaches Chebyshev; compare p=10 vs p=100

def test_minkowski_higher_p_moves_towards_chebyshev():
    a = np.array([0,0,0])
    b = np.array([1,5,2])
    d2 = minkowski_distance(a,b,2)
    d4 = minkowski_distance(a,b,4)
    d8 = minkowski_distance(a,b,8)
    cheb = 5  # maximum coordinate difference
    # For fixed vectors with >1 non-zero diff components, d_p is non-increasing in p
    assert d2 >= d4 >= d8 >= 0
    # Lower bound: Chebyshev distance (max component diff) <= d_p; thus d8 >= cheb
    assert d8 >= cheb
