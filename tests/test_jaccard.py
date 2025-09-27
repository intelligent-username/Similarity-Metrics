from distances import jaccard_distance

# 1. Partial overlap

def test_jaccard_partial_overlap():
    assert jaccard_distance({1,2,3}, {2,3,4}) == 0.5

# 2. Identical sets

def test_jaccard_identical():
    assert jaccard_distance({1,2,3}, {1,2,3}) == 0.0

# 3. Disjoint sets

def test_jaccard_disjoint():
    assert jaccard_distance({'a'}, {'b'}) == 1.0

# 4. Empty vs non-empty

def test_jaccard_empty_vs_nonempty():
    assert jaccard_distance(set(), {1}) == 1.0

# 5. Both empty (define distance 0)

def test_jaccard_both_empty():
    assert jaccard_distance(set(), set()) == 0.0
