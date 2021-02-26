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


def preprocess(X):
    """
    This function takes in training data and applies some preprocessing steps such as scaling and one hot encoding.

    Parameters
    ----------
    X : DataFrame
    Unprocessed data on which to apply preprocessing steps.

    Returns
    -------
    DataFrame
    A DataFrame with appropriate preprocessing steps applied.

    Examples
    --------
    >>> from sklearn.datasets import make_blobs
    >>> X, _ = make_blobs(n_samples=10, centers=3, n_features=2)
    >>> processed_data = preprocess(X)
    
    """

