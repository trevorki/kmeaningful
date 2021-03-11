from kmeaningful import __version__
from kmeaningful.fit_assign import init_centers, assign, measure_dist, \
    calc_centers, fit, fit_assign
import numpy as np
import pytest


def test_version():
    assert __version__ == '0.1.0'


def test_init_centers():
    """Tests that `init_centers()` is working properly"""
    X = np.ones([2, 2])
    k = 3

    # more clusters than points should throw an error
    with pytest.raises(Exception):
        init_centers(X, k)
    # k = 0 clusters should throw an error
    with pytest.raises(Exception):
        init_centers(X, 0)

    # centers are initialized to different points
    centers = init_centers(np.array([[0, 0], [1, 1]]), 2)
    assert np.array_equal(centers, (np.array([[0, 0], [1, 1]]))) \
        or np.array_equal(centers, (np.array([[1, 1], [0, 0]])))
    # the right number of centers are created
    assert init_centers(np.array([[0, 0], [1, 1]]), 2).shape[0] == 2
    # the centers have the same width as the data
    data = np.array([[0, 0, 0], [1, 1, 1]])
    assert init_centers(data, 2).shape[1] == data.shape[1]


def test_assign():
    """Tests that `assign()` is working properly"""

    # different width X and centers should throw an error
    X = np.ones((10, 2))
    centers = np.ones((10, 3))
    with pytest.raises(Exception):
        assign(X, centers)
    # more centers than data points should throw an error
    centers = np.ones((11, 2))
    with pytest.raises(Exception):
        assign(X, centers)

    # check that it returns correct label
    X = np.array([[1], [9]])
    centers = np.array([[0], [10]])
    assert np.array_equal(assign(X, centers), np.array([0, 1]))
    # check that a center can be ignored if no points are nearest to it
    X = np.array([[0], [1], [2], [3]])
    centers = np.array([[1], [10]])
    assert np.array_equal(assign(X, centers), np.array([0, 0, 0, 0]))
    # check that points equidistant to 2 centers are assigned to the first one
    X = np.array([[5], [5]])
    centers = np.array([[0], [10]])
    assert np.array_equal(assign(X, centers), np.array([0, 0]))


def test_measure_dist():
    """Tests that `measure_dist` is working properly"""

    # more centers than data points should throw an error
    X = np.ones((10, 2))
    centers = np.ones((11, 2))
    with pytest.raises(Exception):
        measure_dist(X, centers)

    # check that it calculates 2-d distance correctly
    X = np.array([[0, 0]])
    center = np.array([[3, 4]])
    assert measure_dist(X, center).item() == 5

    # check that it calculates distance between same point is zero
    X = np.array([[0, 0]])
    center = np.array([[0, 0]])
    assert measure_dist(X, center).item() == 0

    # check that for one distance is calculated for each center
    X = np.array([[0, 1], [0, 2], [0, 3]])
    centers = np.array([[0, 0]])
    assert (measure_dist(X, centers) == np.array([[1], [2], [3]])).all()


def test_calc_centers():
    """Tests that `calc_centers` is working properly"""
    X = np.ones((10, 2))
    centers = np.ones((2, 1))
    labels = np.ones(10)

    # different number of data points and labels should throw an error
    with pytest.raises(Exception):
        calc_centers(X, centers, labels)
    # different width for data X and centers should throw an error
    centers = np.ones((2, 2))
    labels = np.ones(9)
    with pytest.raises(Exception):
        calc_centers(X, centers, labels)
    # check one center is correctly calculated
    X = np.array([[-1, 0], [1, 0], [0, 1], [0, -1]])
    # this is used only to determine the number of centers
    centers = np.array([[1000, 1000]])
    labels = np.array([0, 0, 0, 0])
    assert np.array_equal(calc_centers(X, centers, labels), np.array([[0, 0]]))

    # check two centers are correctly calculated
    X = np.array([[-1, 0], [1, 0], [9, 0], [11, 0]])
    # this is used only to determine the number of centers
    centers = np.array([[1000, 1000], [2000, 2000]])
    labels = np.array([0, 0, 1, 1])
    assert np.array_equal(calc_centers(X, centers, labels),
                          np.array([[0, 0], [10, 0]]))

    # check that if one center is the same as another center it gets moved
    X = np.array([[1, 0], [-1, 0], [9, 0], [11, 0], [19, 0], [21, 0]])
    centers = np.array([[0, 0], [100, 0.], [200, 0]])
    labels = np.array([0, 0, 0, 0, 0, 0])
    new_centers = np.array([[10, 0], [-1, 0], [-1, 0]])
    assert np.array_equal(calc_centers(X, centers, labels), new_centers)


def test_fit():
    """Tests that `fit` is working properly"""

    # should throw an error if X contains missing values
    X = np.array([[np.nan], [0]])
    k = 1
    with pytest.raises(Exception):
        fit(X, k)
    # should throw an error if X is not array-like
    X = 1000
    with pytest.raises(Exception):
        fit(X, k)
    # should throw an error if k is not an integer
    X = np.ones((2, 2))
    k = 1.5
    with pytest.raises(Exception):
        fit(X, k)

    # check that four points are correctly assigned to four clusters
    X = np.array([[0, 0], [0, 2], [10, 0], [10, 2]])
    centers = np.array([[0, 0], [0, 2], [10, 0], [10, 2]])
    assert all([fit(X, 4)[i].tolist() in centers.tolist()
                for i in range(centers.shape[0])])

    # check that four points are correctly assigned to two clusters
    X = np.array([[0, 0], [0, 2], [10, 0], [10, 2]])
    centers = np.array([[0, 1], [10, 1]])
    assert all([fit(X, 2)[i].tolist() in centers.tolist()
                for i in range(centers.shape[0])])

    # check that four points are correctly assigned to one cluster
    X = np.array([[0, 0], [0, 2], [10, 0], [10, 2]])
    centers = np.array([[5, 1]])
    assert all([fit(X, 1)[i].tolist() in centers.tolist()
                for i in range(centers.shape[0])])


def test_fit_assign():
    """Tests that `fit_assign` is working properly"""
    X = np.array([[np.nan], [0]])
    k = 2

    # should throw an error if X contains missing values
    with pytest.raises(Exception):
        fit_assign(X, k)
    # should throw an error if X is not array-like
    with pytest.raises(Exception):
        fit_assign(123, k)
    # should throw an error if k is not an integer
    X = np.ones((2, 2))
    with pytest.raises(Exception):
        fit_assign(X, 1.5)

    # check centers and labels for 8 points and 4 clusters
    X = np.array([[-1, 0], [1, 0], [9, 0], [11, 0],
                  [-1, 10], [1, 10], [9, 10], [11, 10]])
    known_centers = np.array([[0., 10.], [10., 0.], [10., 10.], [0., 0.]])
    known_labels = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    centers, labels = fit_assign(X, 4)
    assert all([centers[i].tolist() in known_centers.tolist()
                for i in range(centers.shape[0])])

    # check that each pair of successive points is in same cluster
    assert [labels[i] == labels[i + 1] for i in range(int(len(labels) / 2))]

    # check that there are the correct number of disinct labels
    assert set(labels) == set(known_labels)
