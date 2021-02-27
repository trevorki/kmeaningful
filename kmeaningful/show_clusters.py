def show_clusters(X, centroids, labels = None):
    """
    This function reduces a data set to 2 dimensions using principle component analysis (PCA) and colours clusters of points.
    
    ----------
    X : DataFrame
    Pre-scaled data to train clustering model with.

    centroids: array
    Coordinates of cluster centroids

    labels: list, optional
    Names of clusters, by default None

    Returns
    -------
    plot
    A 2d principle components scatter plot coloured by cluster

    Examples
    --------
    >>> from sklearn.datasets import make_blobs
    >>> X, _ = make_blobs(n_samples=10, centers=3, n_features=2)
    >>> processed_data = preprocess(X)
    >>> optimal_K = find_elbow(processed_data)
    >>> centroids = fit(processed_data, optimal_k)
    >>> show_clusters(processed_data, centroids)
    
    """
