def show_clusters(X, centroids, labels = None):
    """
    This function reduces a data set to 2 dimensions using principle component analysis (PCA) and colours clusters of points.
    
    ----------
    X : DataFrame
    Pre-scaled data to train clustering model with.

    centroids: array
    Coordinates of cluster centroids

    labels: list, optional
    Names of clusters, by default None

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

    pca = PCA(n_components=2)
    principal_comp = pca.fit_transform(data)
    pca_df = pd.DataFrame(data=principal_comp, columns=["pca1", "pca2"])
    pca_df["cluster"] = pd.Series(clusters).apply(str)
    

    plot = alt.Chart(pca_df).mark_point(size=20).encode(
    alt.X('pca1'),
    alt.Y('pca2'),
    color='cluster')
    
    
    
        
    if type(centroids) != type(None):
        try:
            centroids = pd.DataFrame(centroids)
        except ValueError:
            raise ValueError("centroids should be a pandas dataframe or a numpy2darray.")
        #centroids_df = pd.DataFrame(data= pca.transform(centroids), columns=["pca1", "pca2"])
        centroids_df = pd.DataFrame(data= np.array(centroids), columns=["pca1", "pca2"])
        centroids_df["cluster"] = pd.Series(range(centroids.shape[0])).apply(str)

        plot_centroid = alt.Chart(centroids_df).mark_point(size = 100).encode(
            alt.X('pca1'),
            alt.Y('pca2'),
            color='cluster')
        plot = plot + plot_centroid



    return plot