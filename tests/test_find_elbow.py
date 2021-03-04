from kmeaningful.find_elbow import find_elbow
from sklearn.datasets import make_blobs


def test_find_elbow():
    # Make helper data with three distinct clusters
    X, _ = make_blobs(n_samples=10, centers=3, n_features=2)

    # Use the find_elbow function to find the optimal K for the helper data
    optimal_K = find_elbow(X)

    # Make sure the the find_elbow function returns the correct type (int)
    assert isinstance(
        optimal_K, int
    ), f"The type for optimal_K is {type(optimal_K)} but should be int"

    # Make sure the find_elbow function returns the correct number of clusters for the helper data (3)
    assert optimal_K == 3, f"The optimal K should have been 3 but was {optimal_K}"
