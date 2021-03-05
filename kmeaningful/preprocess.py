def preprocess(X):
    """
    This function takes in training data and applies some preprocessing steps such as scaling and one hot encoding.

    Parameters
    ----------
    X : DataFrame
    Unprocessed data on which to apply preprocessing steps.

    Returns
    -------
    Array
    An array with appropriate preprocessing steps applied.

    Examples
    --------
    >>> from sklearn.datasets import make_blobs
    >>> X, _ = make_blobs(n_samples=10, centers=3, n_features=2)
    >>> processed_data = preprocess(X)
    
    """