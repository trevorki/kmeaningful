from kmeaningful.show_clusters import show_clusters
from sklearn.datasets import make_blobs
import pandas as pd


def test_show_clusters():
    X, y = make_blobs(n_samples=20, centers=3, n_features=3, random_state=10)
    a = show_clusters(X, y)

    # Test plot without centroid
    assert a.encoding.x.shorthand == 'pca1', 'x axis should be pca1'
    assert a.encoding.y.shorthand == 'pca2', 'y axis should be pca2'
    assert a.mark.type == 'point', 'marktype should be point'

    # Test plot with centroid
    X_df = pd.DataFrame(X)
    X_df['cluster'] = pd.Series(y).apply(str)
    centroids = X_df.groupby('cluster').mean()
    a = show_clusters(X, y, centroids)

    assert a.layer[0].encoding.x.shorthand == 'pca1', 'x axis should be pca1'
    assert a.layer[0].encoding.y.shorthand == 'pca2', 'y axis should be pca2'
    assert a.layer[0].mark.type == 'point', \
        'mark.type of layer[0] should be point'

    assert a.layer[1].encoding.x.shorthand == 'pca1', 'x axis should be pca1'
    assert a.layer[1].encoding.y.shorthand == 'pca2', 'y axis should be pca2'
    assert a.layer[1].mark.type == 'point', \
        'mark.type of layer[1] should be point'

    print('test_show_clusters() passed')
