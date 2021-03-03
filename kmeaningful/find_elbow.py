def find_elbow(X):
    """
    This function takes in unlabeled, scaled data and performs clustering using the KMeans clustering algorithm values of K up to 10.
    
    It returns the value for K which maximizes the mean silhouette scores for all clusters.

    Parameters
    ----------
    X : DataFrame
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
