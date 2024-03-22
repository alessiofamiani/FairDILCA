# FairDILCA
A fairness aware extension of the Distance Learning for Categorical Attributes (DILCA) framework.

## Repository
This repository is organised as follows:
- **`out`**: this directory is for storing output files (mainly experimental results);
- **`rsc`**: this directory contains datasets and, more in general, input data;
	- `datasets.json`: contains datasets metadata and annotations.
- **`src`**: contains source code.
	- **`algos`**: contains both the DILCA and the FairDILCA implementations;
	- **`utils`**: contains data handling utility functions and metrics computation functions;
	- `exps_clustering.py`: code for clustering experiments;
	- `exps_knn.py`: code for KNN experiments;
	- `exps.py`: code for experiments on the ouputs of the FairDILCA framework;
	- `competitors.py`: code for competitors experiments.

## Usage
	Usage of experiments scripts:
- **Normal** (`exps.py`): `dataset_name n_bins [-max_size int] [-tsne True]`;
- **Clustering** (`exps_clustering.py`): `dataset_name n_bins [-max_size int]`;
- **KNN** (`exps_knn.py`): `dataset_name n_bins [-max_size int]`;
- **Competitors** (`competitors.py`): `competitor exp_type dataset_name n_bins [-max_size int] [-tsne True]`
**Note**: run scripts from the project root folder.
