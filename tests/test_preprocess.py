from kmeaningful import __version__
from kmeaningful.preprocess import preprocess
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
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
    assert expected_output == preprocess(X)
