# kmeaningful 

![](https://github.com/UBC-MDS/kmeaningful/workflows/build/badge.svg) [![codecov](https://codecov.io/gh/UBC-MDS/kmeaningful/branch/main/graph/badge.svg)](https://codecov.io/gh/UBC-MDS/kmeaningful) ![Release](https://github.com/UBC-MDS/kmeaningful/workflows/Release/badge.svg) [![Documentation Status](https://readthedocs.org/projects/kmeaningful/badge/?version=latest)](https://kmeaningful.readthedocs.io/en/latest/?badge=latest)

Have you ever encountered a dataset that seems to have different patterns in it? Have you ever tried to group similar things together in a dataset and to make prediction for a new sample? 

We created `kmeaningful` to help solve such problems. `kmeaningful` is a Python package that uses the k-means algorithm to find and assign labels to clusters, and make prediction on new data points. It also contains functions to help with data preprocessing, hyperparameter tuning and visualizing clusters.

There already exist several packages that implement k-means clustering in Python. Most notably there are the Scikit-learn `sklearn.cluster.KMeans` and SciPy `scipy.cluster.vq.kmeans` implementations. We are not trying to break new ground with `kmeaningful`, but rather to build a simple and lightweight implementation from scratch.

## Installation

```bash
$ pip install -i https://test.pypi.org/simple/kmeaningful
```

## Features

- `preprocess(X)` - Automatic dataset preprocessing: scales numerical features
- `find_elbow(X)` - Automatic hyperparameter tuning to select optimal number of clusters, `k`
- `fit(X)` - finds centroid location for all of the `k` cclusters
- `predict(X, centres)` - assigns each example to a cluster
- `fit_and_predict(X, k)` - Wrapper function that calls `fit(X, k)` and `predict(X, centres)`
- `show_fit(X, k)` - Visualize clusters according to 2d or 3d PCA representation

## Dependencies

- TODO

## Usage

- TODO

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
