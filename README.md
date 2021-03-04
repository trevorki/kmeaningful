# Kmeaningful 

![](https://github.com/UBC-MDS/kmeaningful/workflows/build/badge.svg) [![codecov](https://codecov.io/gh/UBC-MDS/kmeaningful/branch/main/graph/badge.svg)](https://codecov.io/gh/UBC-MDS/kmeaningful) ![Release](https://github.com/UBC-MDS/kmeaningful/workflows/Release/badge.svg) [![Documentation Status](https://readthedocs.org/projects/kmeaningful/badge/?version=latest)](https://kmeaningful.readthedocs.io/en/latest/?badge=latest)

Have you ever encountered a dataset that seems to have different patterns in it? Have you ever tried to group similar things together in a dataset and to assign a new sample based on your findings? 

We created `kmeaningful` to help solve such problems. `kmeaningful` is a Python package that uses the k-means algorithm to find clusters and assign new data points to them. It also contains functions to help with data preprocessing, hyperparameter tuning and visualizing clusters.

## Kmeaningful's Place in the Python Ecosystem

There already exist several packages that implement k-means clustering in Python. Most notably there are the Scikit-learn `sklearn.cluster.KMeans` and SciPy `scipy.cluster.vq.kmeans` implementations. We are not trying to break new ground with `kmeaningful`, but rather to build a simple and lightweight implementation from scratch.

## Installation

```bash
$ pip install -i https://test.pypi.org/simple/kmeaningful
```

## Features

1. `preprocess(X)` - Automatic dataset preprocessing: scales numerical features
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

## Usage

`import kmeaningful as km`

| Task | Function  |
|------------|-----|
| Scale numerical features| `km.preprocess(df)`|
| Find list of centroid points| `km.fit(df, 3)`|
| Assign new data point to cluster| `km.assign(df, array2d)`|
| Find optimal number of cluster| `km.fit_elbow(df)`|
| Visualize data coloured by cluster| `km.show_cluster(df, array2d)`|

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
