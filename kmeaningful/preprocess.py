import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

def preprocess(X):
    """
    This function takes in training data and applies some preprocessing steps such as scaling and one hot encoding.

    Parameters
    ----------
    X : Array-like of shape (n_samples, n_features)
    Unprocessed data on which to apply preprocessing steps, can be in form of DataFrame or Array

    Returns
    -------
    Numpy ndarray of shape (n_samples, n_features_new)
    An array representing the data after appropriate preprocessing steps are applied.

    Examples
    --------
    >>> from sklearn.datasets import make_blobs
    >>> X, _ = make_blobs(n_samples=10, centers=3, n_features=2)
    >>> processed_data = preprocess(X)
    
    """
    
    # Throw error for empty dataframe, alternative is to return empty
    if len(X) < 1:
        raise Exception("Please provide a dataframe X with at least one row as input")
        
    preprocess_pipe = make_pipeline(SimpleImputer(), StandardScaler())
    X_processed = preprocess_pipe.fit_transform(X)

    return X_processed