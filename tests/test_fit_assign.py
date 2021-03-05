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
    """Tests that `init_centers()` is working properly"""
    
    # centers are initialized to different points
    centers = init_centers(np.array([[0, 0], [1, 1]]), 2)
    assert np.array_equal(centers, (np.array([[0, 0], [1, 1]]))) or np.array_equal(centers, (np.array([[1, 1], [0, 0]])))
    
    # the right number of centers are created    
    assert init_centers(np.array([[0, 0], [1, 1]]), 2).shape[0] == 2

    # the centers have the same width as the data
    data = np.array([[0, 0, 0], [1, 1, 1]])
    assert init_centers(data, 2).shape[1] == data.shape[1]


def test_assign(X, centers):
    """Tests that `assign()` is working properly"""
    
    # check that it returns correct label 
    X = np.array([[1], [9]])
    centers = np.array([[0], [10]])    
    assert np.array_equal(assign(X, centers), np.array([0, 1]))
    
    # check that a center can be ignored if no points are nearest to it 
    X = np.array([[0], [1], [2], [3]])
    centers = np.array([[1], [10]])    
    assert np.array_equal(assign(X, centers), np.array([0, 0, 0, 0]))

    # check that points equidistant to 2 centers are assigned to the first one
    X = np.array([[5]]) 
    centers = np.array([[0], [10]])
    assert assign(X, centers).item() == 0

def test_measure_dist():
    """Tests that `measure_dists` is working properly"""

    # check that it calculates 2-d distance correctly
    X = np.array([[0,0]])
    center = np.array([[3,4]])
    assert measure_dist(X, center).item() == 5

    # check that it calculates distance between same point is zero
    X = np.array([[0,0]])
    center = np.array([[0,0]])
    assert measure_dist(X, center).item() == 0

    # check that for one distance is calculated for each center
    X = np.array([[0,0]])
    centers = np.array([[0,1], [0,2], [0,3]])
    assert (measure_dist(X, centers) == np.array([1,2,3])).all()
    

def test_calc_centers(X, centers, labels):
    """Tests that `calc_centers` is working properly"""
    
    # check one center is correctly calculated
    X = np.array([[-1,0], [1,0], [0,1], [0,-1]])
    centers = np.array([[1000,1000]])  #this is used only to determine the number of centers
    labels = np.array([0,0,0,0])
    assert np.array_equal(calc_centers(X,centers,labels), np.array([[0,0]]))

    # check two centers are correctly calculated
    X = np.array([[-1,0], [1,0], [9,0], [11,0]])
    centers = np.array([[1000,1000],[2000,2000]])  #this is used only to determine the number of centers
    labels = np.array([0,0,1,1])
    assert np.array_equal(calc_centers(X,centers,labels), np.array([[0,0], [10,0]]))

    # 


def test_fit(X, k):
    """Tests that `fit` is working properly"""
    
    # check that four points are correctly assigned to four clusters
    X = np.array([[0,0], [0,2], [10,0], [10,2]])
    centers = np.array([[ 0,2], [0, 1], [10,2], [ 10,1]])
    assert fit(X, 4) == centers

    # check that four points are correctly assigned to two clusters
    X = np.array([[0,0], [0,2], [10,0], [10,2]])
    centers = np.array([[0,1], [10,1]])
    assert fit(X, 2) == centers    

    # check that eight points 

# def test_fit_assign(X, k):
    
