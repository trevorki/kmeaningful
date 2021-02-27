def show_fit(X, k):
    """
    This function applies clustering algorithm on a pre-sclaed dataframe and visualize two princple features.
    
    ----------
    X : DataFrame
    Pre-scaled data to train clustering model with.

    k: int
    Number of optimized clusters

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
    >>> show_fit(processed_data, optimal_K)
    
    """