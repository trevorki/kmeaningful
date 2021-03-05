from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans


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
    # Raise error for bad input
    if len(X) < 3:
        raise Exception("Please provide a numpy ndarray with at least three rows")

    scores = dict()
    max_clusters = min(10, X.shape[0] - 1)
    for K in range(2, max_clusters + 1):
        model = KMeans(n_clusters=K)
        model.fit(X)
        labels = model.predict(X)
        sil_score = silhouette_score(X, labels)
        scores[K] = sil_score

    return max(scores, key=lambda k: scores[k])
