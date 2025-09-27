import numpy as np
from distances import cosine_similarity
import math

# 1. Orthogonal vectors => 0

def test_cosine_orthogonal():
    assert cosine_similarity(np.array([1,0]), np.array([0,1])) == 0.0

# 2. Same direction => 1

def test_cosine_same_direction():
    import math
    assert math.isclose(cosine_similarity(np.array([1,1]), np.array([2,2])), 1.0, rel_tol=1e-12)

# 3. Opposite direction => -1

def test_cosine_opposite():
    assert math.isclose(cosine_similarity(np.array([1,1]), np.array([-1,-1])), -1.0)

# 4. Normalization invariance (scaling one vector doesn't change result)

def test_cosine_scale_invariance():
    v1 = np.array([3,4,0])
    v2 = np.array([6,8,0])
    assert cosine_similarity(v1,v2) == 1.0

# 5. Mixed components (known value ~0.70710678)

def test_cosine_known_angle():
    val = cosine_similarity(np.array([1,1]), np.array([1,0]))
    assert math.isclose(val, 0.70710678, rel_tol=1e-8, abs_tol=1e-8)
