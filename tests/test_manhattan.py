import numpy as np
from distances import manhattan_distance

# 1. Basic 2D example
def test_manhattan_basic():
    assert manhattan_distance(np.array([0,0]), np.array([3,4])) == 7

# 2. 3D example from doc-like usage
def test_manhattan_3d():
    a = np.array([1,2,3])
    b = np.array([-4,-5,-6])
    assert manhattan_distance(a,b) == 21

# 3. Zero distance

def test_manhattan_zero():
    v = np.array([10, -5, 2])
    assert manhattan_distance(v,v) == 0

# 4. Mixed sign stability
def test_manhattan_mixed_signs():
    a = np.array([5,-3,9])
    b = np.array([-2,4,-1])
    assert manhattan_distance(a,b) == (abs(5+2)+abs(-3-4)+abs(9+1))

# 5. Large values (linear growth)
def test_manhattan_large_values():
    a = np.array([1e6, -1e6])
    b = np.array([1e6 + 3, -1e6 + 4])
    assert manhattan_distance(a,b) == 7
