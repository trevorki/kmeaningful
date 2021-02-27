def fit_predict(X, k):
    """
    This function takes in unlabeled, scaled data and performs clustering using the KMeans clustering algorithm.
    
    It returns a list of the cluster labels for each point.

    Parameters
    ----------
    X : DataFrame
    Pre-scaled data to train clustering model with.

    k : int
    The number of clusters to use for Kmeans.

    Returns
    -------
    list
    A list containing the cluster label for every example (row) in X.

    Examples
    --------
    >>> from sklearn.datasets import make_blobs
    >>> X, _ = make_blobs(n_samples=10, centers=3, n_features=2)
    >>> processed_data = preprocess(X)
    >>> cluster_assignments = fit_predict(processed_data, 3)    
    """