from kmeaningful.show_cluster import show_cluster
from sklearn.datasets import make_blobs
import altair as alt

def test_show_clusters():
  X, y = make_blobs(n_samples=10, centers=3, n_features=3, random_state=10)
  a = show_clusters(X, y)

  # Need more tests for different layers
  assert a.encoding.x.shorthand == 'pca1', a.encoding.x.shorthand
  assert a.encoding.y.shorthand =='pca2', a.encoding.y.shorthand
  assert a.mark == 'point', a.mark