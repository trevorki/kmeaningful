def fit(X, k):
    """
    This function takes in unlabeled, scaled data and performs clustering using the KMeans clustering algorithm.
    
    It returns an array of the coordinates of the k cluster centers.

    Parameters
    ----------
    X : DataFrame
    Pre-scaled data to train clustering model with.

    k : int
    The number of clusters to use for Kmeans.

    Returns
    -------
    array
    A (k,d) array of the center locations for each cluster where d = number of dimensions

    Examples
    --------
    >>> from sklearn.datasets import make_blobs
    >>> X, _ = make_blobs(n_samples=10, centers=3, n_features=2)
    >>> processed_data = preprocess(X)
    >>> centers = fit(processed_data, 3)
    
    """
