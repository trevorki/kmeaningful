from kmeaningful import __version__
from kmeaningful.fit_assign import init_centers, assign, measure_dist, calc_centers, fit, fit_assign, plot_clusters
import pandas as pd
import numpy as np
import random
import pytest

X = np.array([[-1,0], [1,0], [9,0], [11,0], [-1,10], [1,10], [9,10], [11,10]])
centers = np.array([[ 0., 10.], [10.,  0.], [10., 10.], [ 0.,  0.]])
# labels = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]

def test_version():
    assert __version__ == '0.1.0'

def test_init_centers():
    """check that init_centers is working properly"""
    
    # centers are initialized to different points
    centers = init_centers(np.array([[0, 0], [1, 1]]), 2)
    assert np.array_equal(centers, (np.array([[0, 0], [1, 1]]))) or np.array_equal(centers, (np.array([[1, 1], [0, 0]])))
    
    # the right number of centers are created    
    assert init_centers(np.array([[0, 0], [1, 1]]), 2).shape[0] == 2

    # the centers have the same width as the data
    data = np.array([[0, 0, 0], [1, 1, 1]])
    assert init_centers(data, 2).shape[1] == data.shape[1]


# def test_assign(X, centers):
    


# def test_measure_dist(X, centers):
    

# def test_calc_centers(X, centers, labels):
    

# def test_fit(X, k):
    
    

# def test_fit_assign(X, k):
    
