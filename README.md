# Kmeaningful 

![](https://github.com/UBC-MDS/kmeaningful/workflows/build/badge.svg) [![codecov](https://codecov.io/gh/UBC-MDS/kmeaningful/branch/main/graph/badge.svg)](https://codecov.io/gh/UBC-MDS/kmeaningful) ![Release](https://github.com/UBC-MDS/kmeaningful/workflows/Release/badge.svg) [![Documentation Status](https://readthedocs.org/projects/kmeaningful/badge/?version=latest)](https://kmeaningful.readthedocs.io/en/latest/?badge=latest)

Have you ever encountered a dataset that seems to have different patterns in it? Have you ever tried to group similar things together in a dataset and to assign a new sample based on your findings? 

We created `kmeaningful` to help solve such problems. `kmeaningful` is a Python package that uses the k-means algorithm to find clusters and assign new data points to them. It also contains functions to help with data preprocessing, hyperparameter tuning and visualizing clusters.

## Kmeaningful's Place in the Python Ecosystem

There already exist several packages that implement k-means clustering in Python. Most notably there are the Scikit-learn `sklearn.cluster.KMeans` and SciPy `scipy.cluster.vq.kmeans` implementations. We are not trying to break new ground with `kmeaningful`, but rather to build a simple and lightweight implementation from scratch.

## Installation

```bash
$ pip3 install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple kmeaningful==0.2.0
```

## Features

1. `preprocess(X)` - Automatic dataset preprocessing using scaling or one-hot-encoding based on column type
2. `find_elbow(X)` - Automatic hyperparameter tuning to select optimal number of clusters, `k`
3. `fit_assign(X, k)` - Wrapper function that calls `fit(X, k)` and `assign(X, centres)`
    - `fit(X, k)` - finds centroid location for all of the `k` clusters
    - `assign(X, centres)` - assigns each example to a cluster
4. `show_clusters(X, centres)` - Visualize clusters according to 2d PCA representation

## Dependencies

- Python 3.8.0 and Python packages:
    - pandas==^1.2.3
    - pytest==6.2.2 
    - sklearn==^0.0 
    - altair==^4.1.0
    - flake8==^3.8.4 
    - codecov==^2.1.11
    - python-semantic-release==^7.15.0

## Usage

`from kmeaningful import preprocess, fit_assign, find_elbow, show_clusters`

| Task | Function  |
|------------|-----|
| Scale numeric features and use OHE on categorical features| `preprocess.preprocess(df)`|
| Find optimal number of cluster| `find_elbow.find_elbow(df)`|
| Find list of centroid points| `fit_assign.fit(df, 3)`|
| Assign new data point to cluster| `fit_assign.assign(df, array2d)`|
| Visualize data coloured by cluster| `show_clusters.show_cluster(df, array2d)`|

## Example

``` py

from kmeaningful import preprocess, fit_assign, find_elbow, show_clusters

example_df = [[1,2,1],[1,3,3],[2,3,4]]
processed_data = preprocess.preprocess(example_df)
optimal_K = find_elbow.find_elbow(processed_data)
centers, labels = fit_assign.fit_assign(processed_data, optimal_K)
show_clusters.show_clusters(processed_data, labels, centers)

```


## Documentation

The official documentation is hosted on Read the Docs: https://kmeaningful.readthedocs.io/en/latest/

## Contributors
This project was created by DSCI 524 Group 16: 
- Yihong (Hazel) Jiang
- Mike Lynch
- Trevor Kinsey
- Sasha Babicki

We welcome and recognize all contributions. You can see a list of current contributors in the [contributors tab](https://github.com/UBC-MDS/kmeaningful/graphs/contributors).

### Credits

This package was created with Cookiecutter and the UBC-MDS/cookiecutter-ubc-mds project template, modified from the [pyOpenSci/cookiecutter-pyopensci](https://github.com/pyOpenSci/cookiecutter-pyopensci) project template and the [audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage).
