import numpy as np
from distances import gower_distance

# Helper to build ranges quickly

def test_gower_all_numeric_max_difference():
    a = np.array([1,2,3])
    b = np.array([4,5,6])
    ranges = np.array([3,3,3])
    assert gower_distance(a,b,ranges) == 1.0


def test_gower_all_numeric_partial_difference():
    a = np.array([1,2,3])
    b = np.array([1,5,6])
    ranges = np.array([0,3,3])  # first feature same => range 0 yields 0 contribution
    # Differences: feature2: |2-5|/3=1; feature3: |3-6|/3=1 => (0+1+1)/2 considered? weight counts non-empty features.
    # weight =3 because both points have values for each feature even if range 0? Implementation sets dist 0 when range=0 numeric.
    # So total distance (0+1+1)/3 = 2/3
    assert np.isclose(gower_distance(a,b,ranges), 2/3)


def test_gower_mixed_all_different():
    a = np.array([1,'cat',3], dtype=object)
    b = np.array([4,'dog',6], dtype=object)
    ranges = np.array([3,0,3], dtype=float)
    assert gower_distance(a,b,ranges) == 1.0


def test_gower_mixed_identical():
    a = np.array(['a','b','c'], dtype=object)
    b = np.array(['a','b','c'], dtype=object)
    ranges = np.array([0,0,0], dtype=float)
    assert gower_distance(a,b,ranges) == 0.0


def test_gower_partial_categorical_difference():
    a = np.array([1,'x',2], dtype=object)
    b = np.array([2,'x',2], dtype=object)
    ranges = np.array([1,0,0], dtype=float)
    # Feature1 difference: |1-2|/1 =1 ; feature2 same categorical =>0 ; feature3 same numeric range=0 =>0
    # distance=1/3
    assert np.isclose(gower_distance(a,b,ranges), 1/3)
