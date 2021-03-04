



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
    n = X.shape[0]
    dimensions = X.shape[1]
    centers = np.zeros((k, dimensions))
    ind = []

    # pick 1st center at random
    ind.append(random.randint(0,n-1))
    centers[0,] = X[ind[0]]

    #find rest of centers
    for kk in range(1, k):
        dists_sq = measure_dist(X, centers[0:kk])**2
        for i in ind:                                    # set distance between centers to 0
            dists_sq[i] = np.zeros((1,dists_sq.shape[1]))
        dists_sq[dists_sq == 0] = np.inf                 # replace 0 with inf
        dists_sq = dists_sq.min(axis = 1)                # select minimum distance in row
        dists_sq[dists_sq == np.inf] = 0                 # replace inf with 0 again
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
    list
    A list with the cluster assignments for the data.
    
    Examples
    --------
    >>> from sklearn.datasets import make_blobs
    >>> X, _ = make_blobs(n_samples=10, centers=3, n_features=2)
    >>> centers = fit(X, 3)
    >>> cluster_assignments = predict(X, centers)
    """    
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
    The locations of the cluster centers. Dimensions: (k,d)
    
    labels: list
    The assigned cluster for each data point in X. Length: n

    Returns
    -------
    array
    The distances from each point to each center. Dimensions: (n, k)
    
    """
    n = X.shape[0]
    d = X.shape[1]
    k = centers.shape[0]
    
    new_centers = np.zeros((k,d))
    for kk in range(k):
        new_center = [np.mean(X[labels == kk][:,dd]) for dd in range(d)]
        if np.isnan(np.sum(new_center)) == False:
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
    A (k,d) array of the center locations for each cluster where d = number of dimensions.
    
    Examples
    --------
    >>> from sklearn.datasets import make_blobs
    >>> X, _ = make_blobs(n_samples=10, centers=3, n_features=2)
    >>> centers = fit(X, 3)
    
    """
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
    >>> cluster_assignments = fit_predict(X, 3)    
    """
    centers = fit(X, k)
    labels = assign(X, centers)
       
    return centers, labels



# plotting functions
def plot_clusters(X, centers, labels = None, title = ""):
    """
    makes a 2d plot of points in `X`, the cluster centers in `clusters`,
    coloured by nearest cluster contained in `labels`
    """
    colours = range(len(centers))
    plt.scatter(X[:, 0], X[:, 1], s = 10, c = labels, alpha = 0.4)
    plt.scatter(centers[:, 0], centers[:, 1], s = 400, 
                    marker = '*', c = colours)
    plt.title(title);
    plt.show()
    return