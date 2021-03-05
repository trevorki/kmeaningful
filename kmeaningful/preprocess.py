import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer
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
    
    # Throw error if input is not array-like
    try: 
        df = pd.DataFrame(X)
    except:
        raise Exception("Input format not accepted")
    
    numeric_features = df.select_dtypes("number").columns
    categorical_features = df.select_dtypes("object").columns
    
    numeric_transformer = make_pipeline(
        SimpleImputer(),
        StandardScaler()
    )
    
    categorical_transformer = make_pipeline(
        OneHotEncoder(handle_unknown="ignore")
    )
    
    preprocessor = make_column_transformer(
        (numeric_transformer, numeric_features), 
        (categorical_transformer, categorical_features)
    )
    
    X_processed = preprocessor.fit_transform(X)

    return X_processed