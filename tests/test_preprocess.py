from kmeaningful import __version__
from kmeaningful import preprocess
import pandas as pd

def test_version():
    assert __version__ == '0.1.0'

def test_preprocess():
    """ Performs tests for preprocess function """

    X = pd.DataFrame({})
    assert preprocess.preprocess(X) == None
