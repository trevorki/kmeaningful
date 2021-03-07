import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
import altair as alt

def show_clusters(data, clusters, centroids = None):
    """
    This function reduces a data set to 2 dimensions using principle component analysis (PCA) and colours clusters of points.
    Parameters


    ----------
    data : DataFrame
    Scaled data

    clusters : list, pandas Series
    corresponding cluster for X

    centroids: 2d array
    Coordinates of cluster centroids


    Returns
    -------
    plot
    A 2d principle components scatter plot coloured by cluster
    Examples
    --------
    >>> from sklearn.datasets import make_blobs
    >>> X, _ = make_blobs(n_samples=10, centers=3, n_features=2)
    >>> processed_data = preprocess(X)
    >>> optimal_K = find_elbow(processed_data)
    >>> centroids = fit(processed_data, optimal_k)
    >>> show_clusters(processed_data, centroids)

    
    """
    
    # Exception handling
    try:
        data = pd.DataFrame(data)
    except ValueError:
        raise ValueError("data should be a pandas dataframe or a numpy2darray.")
        
    try: 
        clusters = pd.Series(clusters)
    except ValueError:
        raise ValueError("clusters should be a list of numbers, a pandas series or a numpy1darray.")
        
    if data.shape[0] != clusters.shape[0]:
        raise ValueError("data should have the same number of rows as clusters")

    # Plot without centroid
    pca = PCA(n_components=2)
    principal_comp = pca.fit_transform(data)
    pca_df = pd.DataFrame(data=principal_comp, columns=["pca1", "pca2"])
    pca_df["cluster"] = pd.Series(clusters).apply(str)
    

    plot = alt.Chart(pca_df).mark_point(size=20).encode(
    alt.X('pca1'),
    alt.Y('pca2'),
    color='cluster')
    
    
    
    # Exception handling for plot with centroid    
    if type(centroids) != type(None):
        try:
            centroids = pd.DataFrame(centroids)
        except ValueError:
            raise ValueError("centroids should be a pandas dataframe or a numpy2darray.")
        centroids_df = pd.DataFrame(data= pca.transform(centroids), columns=["pca1", "pca2"])
        #centroids_df = pd.DataFrame(data= np.array(centroids), columns=["pca1", "pca2"])
        centroids_df["cluster"] = pd.Series(range(centroids.shape[0])).apply(str)

        plot_centroid = alt.Chart(centroids_df).mark_point(filled=True, shape='cross', size = 150,opacity=1).encode(
            alt.X('pca1'),
            alt.Y('pca2'),
            color='cluster')
        plot = plot + plot_centroid



    return plot