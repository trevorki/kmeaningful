from kmeaningful import __version__
from kmeaningful.preprocess import preprocess

from sklearn.datasets import make_blobs

import pandas as pd
import numpy as np

import pytest

def test_version():
    assert __version__ == '0.1.0'

def test_preprocess():
    """ Performs tests for preprocess function """

    # empty dataframe should throw exception
    X = pd.DataFrame({})
    assert pytest.raises(Exception, preprocess, X)
    
    # dataframe with one col and one row 0 should be the same scaled or not
    X = pd.DataFrame({"col1":[0]})
    expected_output = X.to_numpy().astype(float)
    assert (expected_output == preprocess(X)).all()
    
    # return type of processed data should be numpy.ndarray
    assert type(preprocess(X)) is np.ndarray

    # dataframe with two cols with same values should be [[0., 0.]]
    X = pd.DataFrame({"col1":[1], "col2":[1]})
    expected_output = np.array([[0., 0.]])
    assert (expected_output == preprocess(X)).all()

    # scaling is working as expected
    X, _ = make_blobs(n_samples=10, centers=3, n_features=2)
    processed_data = preprocess(X)
    assert X.max() >= processed_data.max()
    assert X.min() <= processed_data.min()

    # imputation is working as expected
    mask = np.random.choice([True, False], size=X.shape) 
    X[mask] = None  # set entries as None at random
    assert np.isnan(X).any()  # check that test code working
    assert not np.isnan(preprocess(X)).any()  # result should not have nans

    # handle missing data with imputation
    X = pd.DataFrame({"col1":[None], "col2": [1]})
    expected_output = np.array([[0., 0.]])  # should be filled in with mean and then scaled
    assert (preprocess(X) == expected_output).all()

    # reject when all data missing
    X = [[None, None], [None, None]]
    assert pytest.raises(Exception, preprocess, X)

    # use correct one-hot-encoding for categorical data
    X = [["neutral", "large"],["neutral", "medium"]]
    expected_output = np.array([[1., 1., 0.], [1., 0., 1.]])
    assert (preprocess(X) == expected_output).all()
    
    # possibly update in future to use BOW encoding for other non-numeric data 
    # e.g. length > 20 and no repeated words in col
    
    
    
