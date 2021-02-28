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
    
    return predict(X, fit(X, k))

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
    arrag
    A (k,d) array of the center locations for each cluster where d = number of dimensions.

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

    centers : array
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
