from sklearn.metrics import silhouette_score
from kmeaningful.fit_assign import fit_assign
import numpy as np


def find_elbow(X):
    """
    This function takes in unlabeled, scaled data and performs clustering using the KMeans clustering algorithm values of K up to the min(10, n_samples - 1).
    
    It returns the value for K which maximizes the mean silhouette scores for all clusters.

    Parameters
    ----------
    X : numpy ndarray
    Pre-scaled data to train clustering model with.

    Returns
    -------
    int
    The value for K which maximizes the mean silhouette scores for all clusters.

    Examples
    --------
    >>> from sklearn.datasets import make_blobs
    >>> X, _ = make_blobs(n_samples=10, centers=3, n_features=2)
    >>> processed_data = preprocess(X)
    >>> optimal_K = find_elbow(processed_data)
    
    """
    # Raise exception for bad input
    if not type(X) == np.ndarray:
        raise Exception(
            f"Please provide a numpy ndarray as input. Input type detected: {type(X)}"
        )

    # This check is needed because silhouette score is only defined for 1 < n_labels < n_samples
    if len(X) < 3:
        raise Exception("Please provide a numpy ndarray with at least three rows")

    # Calculate the max possible value for K given the input data
    max_clusters = min(10, X.shape[0] - 1)

    # Calculate the mean silhouette score for each K
    scores = dict()
    for K in range(2, max_clusters + 1):
        _, labels = fit_assign(X, K)
        sil_score = silhouette_score(X, labels)
        scores[K] = sil_score

    # Return value for K which results in greatest silhouette score
    return max(scores, key=lambda k: scores[k])
