import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

def init_centers(X, k):
    """
    This function chooses initial cluster locations using Kmeans++

    Parameters
    ----------
    X : array
    Data used to find clusters.  Dimensions: (n,d)

    k : int
    The desired number of clusters .

    Returns
    -------
    array
    Array containing the initial coordinates of the k clusters

    Examples
    --------
    >>> from sklearn.datasets import make_blobs
    >>> X, _ = make_blobs(n_samples=10, centers=3, n_features=2)
    >>> intial_centers = init(X, 3)    
    """

    # Throw error if k > number of data points
    if k > X.shape[0]:
        raise Exception("Number of clusters must be less than number of data points")
     # Throw error if k is negative
    if k <= 0:
        raise Exception("Number of clusters must be a positive integer")
    # Throw error if X is empty
    if X.shape[0] == 0:
        raise Exception("Input data must have at least one row")

    n = X.shape[0]
    dimensions = X.shape[1]
    centers = np.zeros((k, dimensions))
    ind = []             #indeces of existing centers

    # pick 1st center at random
    ind.append(random.randint(0,n-1))
    centers[0,] = X[ind[0]]

    #find rest of centers
    for kk in range(1, k):
        dists_sq = measure_dist(X, centers[0:kk])**2     # measure distance from every point to current center
        for i in ind:                                    # set distance between existing centers to 0
            dists_sq[i] = np.zeros((1,dists_sq.shape[1]))
        dists_sq[dists_sq == 0] = np.inf                 # replace 0 with inf so they don't get selected
        dists_sq = dists_sq.min(axis = 1)                # select minimum distance in row
        dists_sq[dists_sq == np.inf] = 0                 # replace inf with 0 again to make probability of selecting existing center zero
        probs = (dists_sq / np.sum(dists_sq)).tolist()   # probability prop to dist_sq
        ind.append(np.random.choice(range(len(probs)), size=1, p=probs)) #select point at random
        centers[kk,] = X[ind[-1]]
    return centers


def assign(X, centers):
    """
    Assigns data to clusters based on Euclidean distance to the nearest centroid. 

    Parameters
    ----------
    X : array
    Data for cluster assignment.  Dimensions: (n,d)

    centers : array
    The locations of the cluster centers.

    Returns
    -------
    array
    The cluster assignments for the data.
    
    Examples
    --------
    >>> from sklearn.datasets import make_blobs
    >>> X, _ = make_blobs(n_samples=10, centers=3, n_features=2)
    >>> centers = fit(X, 3)
    >>> cluster_assignments = predict(X, centers)
    """ 

    # Throw error if X and centers have different widths
    if X.shape[1] != centers.shape[1]:
        raise Exception("`X` and `centers` must have the same width")
    # Throw error if there are more centers than data points
    if X.shape[0] <= centers.shape[0]:
        raise Exception("There are more centers than data points")
    # Throw error if there are no centers
    if centers.shape[0] == 1:
        raise Exception("There are no centers defined")


    n = X.shape[0]
    k = centers.shape[0]
    labels = np.zeros(n, dtype = int)
    distances = measure_dist(X, centers)
    for nn in range(n):
        labels[nn] = np.argmin(distances[nn])
    return labels


def measure_dist(X, centers):
    """
    Measures the euclidean distance between each row (point) in `X`,
    and each row (cluster centre) in `centers`
    
    Parameters
    ----------
    X : array
    Data for cluster assignment. Dimensions: (n,d)

    centers : array
    The locations of the cluster centers. Dimensions: (k,d)

    Returns
    -------
    array
    The distances from each point to each center. Dimensions: (n, k)
    
    Examples
    --------
    >>> from sklearn.datasets import make_blobs
    >>> X, _ = make_blobs(n_samples=10, centers=3, n_features=2)
    >>> centers = fit(X, 3)
    >>> distances = predict(X, centers)
    """
    # Throw error if X and centers have different widths
    if X.shape[1] != centers.shape[1]:
        raise Exception("`X` and `centers` must have the same width")
    # Throw error if there are more centers than data points
    if X.shape[0] <= centers.shape[0]:
        raise Exception("There are more centers than data points")
    # Throw error if there are no centers
    if centers.shape[0] == 0:
        raise Exception("There are no centers defined")


    n = X.shape[0]
    k = centers.shape[0]
    distances = np.zeros((n,k))
    for kk in range(k):  
        for nn in range(n):
            pt = X[nn,]
            cent = centers[kk,]          
            distances[nn,kk] = np.sqrt(np.sum((pt - cent)**2))
    return distances


def calc_centers(X, centers, labels):
    """
    Calculates the coordinates of the centroid of each cluster
    
    Parameters
    ----------
    X : array
    Data for cluster assignment. Dimensions: (n,d)

    centers : array
    The locations of the cluster centers. Dimensions: (k,d). Used only to determine number of clusters
    
    labels: array
    The assigned cluster for each data point in X. Dimensions: (n,)

    Returns
    -------
    array
    A (k,d) array of the center locations for each cluster.
    
    """
    #  Throw error if `X` and `labels` have different lengths
    if X.shape[0] != len(labels):
        raise Exception("The number of labels is different from the number of points")
    # Throw error if `X` and `centers` have different widths
    if X.shape[1] != centers.shape[1]:
        raise Exception("`X` and `centers` must have the same width")

    n = X.shape[0]
    d = X.shape[1]
    k = centers.shape[0]
    
    new_centers = np.zeros((k,d))
    for kk in range(k):
        new_center = [np.mean(X[labels == kk][:,dd]) for dd in range(d)] # 
        if np.isnan(np.sum(new_center)) == False:   # If there 
            new_centers[kk] = new_center
        else:                                 # if there is no nearest point, assign to farthest point
            dists = measure_dist(X, centers[kk])
            new_centers[kk] = X[np.argmax(dists)//d,]  
            print(f"centre {kk} has no nearest points, reassign to {new_centers[kk]}")
            
    return new_centers

def fit(X, k):
    """
    This function takes in unlabeled, scaled data and performs clustering using the KMeans clustering algorithm.
    
    Parameters
    ----------
    X : array
    Data to train clustering model with.  Dimensions: (n,d)

    k : int
    The number of clusters to use for Kmeans.

    Returns
    -------
    array
    A (k,d) array of the center locations for each cluster.
    
    Examples
    --------
    >>> from sklearn.datasets import make_blobs
    >>> X, _ = make_blobs(n_samples=10, centers=3, n_features=2)
    >>> centers = fit(X, 3)
    
    """
    # Throw error if X contains nan values
    if np.isnan(X).any():
        raise Exception("Array contains missing data")
    # Throw error if X is not array-like
    try: 
        df = pd.DataFrame(X)
    except:
        raise Exception("Input format not accepted")
    #  Throw error if k is not an integer
    if isinstance(k, int) != True:
        raise Exception("k must be an integer")

    # initialize cluster centers and assign points to clusters
    centers = init_centers(X, k)
    i = 0    # iteration counter
    
    # first iteration
    labels = assign(X, centers) # assign cluster label based on closest center
    new_centers = calc_centers(X, centers, labels)
    new_labels = assign(X, centers)
    # plot_clusters(X, centers, labels = new_labels, title = f"Initial clusters")

    i += 1               #initialize iteration counter

    #subsequent iterations
    while((np.sum(new_centers - centers)) and (i < 20)):
        centers = new_centers
        labels = new_labels
        new_labels = assign(X, centers) # assign cluster label based on closest center
        new_centers = calc_centers(X, centers, new_labels)

        i+=1

    return new_centers
    

def fit_assign(X, k):
    """
    This function takes in data and performs clustering using the KMeans clustering algorithm.

    Parameters
    ----------
    X : array
    Pre-scaled data to train clustering model with. Dimensions: (n,d)

    k : int
    The number of clusters to use for Kmeans.
    
    Returns
    -------
    array
    The coordinates of the cluster centers
    
    list
    A list containing the cluster label for every example (row) in X.

    Examples
    --------
    >>> from sklearn.datasets import make_blobs
    >>> X, _ = make_blobs(n_samples=10, centers=3, n_features=2)
    >>> centers, labels = fit_assign(X, 3, centers)
    """
    # Throw error if X contains nan values
    if np.isnan(X).any():
        raise Exception("Array contains missing data")
    # Throw error if X is not array-like
    try: 
        df = pd.DataFrame(X)
    except:
        raise Exception("Input format not accepted")
    #  Throw error if k is not an integer
    if isinstance(k, int):
        raise Exception("k must be an integer")


    centers = fit(X, k)
    labels = assign(X, centers)
       
    return centers, labels
