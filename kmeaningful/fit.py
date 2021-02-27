def fit(X, k):
    """
    This function takes in unlabeled, scaled data and performs clustering using the KMeans clustering algorithm.
    
    It returns a length k list of the locations of cluster centers.

    Parameters
    ----------
    X : DataFrame
    Pre-scaled data to train clustering model with.

    k : int
    The number of clusters to use for Kmeans.

    Returns
    -------
    list
    A list size k of the center locations for each cluster.

    Examples
    --------
    >>> from sklearn.datasets import make_blobs
    >>> X, _ = make_blobs(n_samples=10, centers=3, n_features=2)
    >>> processed_data = preprocess(X)
    >>> centers = fit(processed_data, 3)
    
    """
