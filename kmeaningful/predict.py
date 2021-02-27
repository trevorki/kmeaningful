def predict(X, centers):
    """
    Assigns data to clusters based on Euclidean distance to the nearest centroid. 

    Parameters
    ----------
    X : DataFrame
    Pre-scaled data for cluster assignment.

    centers : list
    The locations of the cluster centers.

    Returns
    -------
    list
    A list with the cluster assignments for the data.
    
    Examples
    --------
    >>> from sklearn.datasets import make_blobs
    >>> X, _ = make_blobs(n_samples=10, centers=3, n_features=2)
    >>> processed_data = preprocess(X)
    >>> centers = fit(processed_data, 3)
    >>> cluster_assignments = predict(processed_data, centers)
    """
