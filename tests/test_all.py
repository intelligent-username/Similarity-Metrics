# Aggregator to ensure all per-metric test modules import cleanly.
# Pytest will still discover individual tests, but this file can be used to
# verify import-time errors early or run `pytest -k test_all` quickly.

import importlib
import sys

MODULES = [
    'tests.test_euclidean',
    'tests.test_manhattan',
    'tests.test_minkowski',
    'tests.test_cosine',
    'tests.test_gower',
    'tests.test_jaccard',
]

def test_import_all_modules():
    for m in MODULES:
        importlib.import_module(m)
        assert m in sys.modules
