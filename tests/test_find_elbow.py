from kmeaningful.find_elbow import find_elbow
from kmeaningful.preprocess import preprocess
from sklearn.datasets import make_blobs
import numpy as np
import pytest


def test_find_elbow():

    # empty array should raise exception
    empty = np.ndarray(0)
    assert pytest.raises(Exception, find_elbow, empty)

    # 1 row array should raise exception
    single_row = np.ones((1, 2))
    assert pytest.raises(Exception, find_elbow, single_row)

    # Make low dimensional helper data with three distinct clusters
    X, _ = make_blobs(n_samples=10, centers=3, n_features=2, random_state=1234)
    X_scaled = preprocess(X)

    # Use the find_elbow function to find the optimal K for the helper data
    optimal_K = find_elbow(X_scaled)

    # Make sure the the find_elbow function returns the correct type (int)
    assert isinstance(
        optimal_K, int
    ), f"The type for optimal_K is {type(optimal_K)} but should be int"

    # Make sure the find_elbow function returns the correct number of clusters for the helper data (3)
    assert optimal_K == 3, f"The optimal K should have been 3 but was {optimal_K}"
