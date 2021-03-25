# Introduction to Run Source Code
The code is also provided at 
https://github.gatech.edu/xzhou94/ML_Fall2020/tree/master/unsupervised_learning

## Unsupervised Learning
### Setup the Environment
Follow the steps to set up your environment:
* Make sure python 3 is installed.
* Install `pyenv` toolkit for python and site package version control.
* Create and enter the virtual environment using following commend in the bash. 
Make sure the `virtualenv` is under python3 :
```commandline
pyenv virtualenv 3.7.7 gtml
pyenv local gtml
```

* Go under the `supervised_learning/` folder to install the required packages.
```commandline
cd unsupervised_learning/
pip install -r requirements.txt
``` 

### Run Code
Make sure your are in the `gtml` virtual environment and your environment is well set up
before running the code. `campaign_marketing` is a relatively smaller dataset, and the 
complete reproducing take couple of minutes. However, please expect a longer time to 
reproduce the second dataset `university_recommendation`. 
This instruction will lead you to save all the 
key results of the training in order to 
reproduce the plots and offline metrics in my report.   
Please create folders under `unsupervised_learning/`.
```commandline
mkdir logdir
mkdir results
mkdir plots
```

#### Preliminary Analysis and Visualization
Run the following code will generate k-means and EM clustering for selected 
original dataset. The results will be stored as pickle in the  `results/` folder. 

##### Clustering Algorithms
```commandline
# for dataset: campaign_marketing
python clustering.py campaign_marketing


# for dataset: university_recommendation
python clustering.py university_recommendation
```
The clustering plots in the report can be reproduced by running following commendlines.
```commandline
# for dataset: campaign_marketing
python visualize_clusters.py campaign_marketing


# for dataset: university_recommendation
python visualize_clusters.py university_recommendation
```

##### Feature Transformation/Selection Algorithms
```commandline
# for dataset: campaign_marketing
python dimensionality_reduction.py campaign_marketing


# for dataset: university_recommendation
python dimensionality_reduction.py university_recommendation
```
The transformation plots in the report can be reproduced by running following commendlines.
```commandline
# for dataset: campaign_marketing
python visualize_dim_reduction.py campaign_marketing


# for dataset: university_recommendation
python visualize_dim_reduction.py university_recommendation
```


#### Clustering on Reduced Dimensions
In this experiment, four feature transformations are conducted before two clustering algorithms, and they 
are applied on CM and UR datasets. In total, there are 16 combinations of subexperiments.
To reuse the code at largest extent, please use the following commandlines to generate the results of the 
16 experiments. Note all the useful metrics are stored in the `results/` folder with the name 
`{dataset_name}_{step}_{clustering_method}_{reduce_method}.pkl` 
(eg: `campaign_marketing_clustering_on_reduced_kmeans_dt.pkl`). To conduct this experiment, 
there are parameters required to be predefined, such as the optimal `n_component` of PCA and etc.
Those parameters are listed as a python method with `dataset_name` as the parameter.
Feel free to tweak the parameters to explore more combinations. 
Following is the place that to change those parameters.
```python
# within clustering_on_reduced_dimension.py
def define_params(dataset_name):
    if dataset_name == 'campaign_marketing':
        return {
            'pca': {'n_components': 20},
            'ica': {'n_components': 8},
            'sparse_rca': {'n_components': 45},
            'dt': {'cutoff': 0.8}
        }
    elif dataset_name == 'university_recommendation':
        return {
            'pca': {'n_components': 200},
            'ica': {'n_components': 90},
            'sparse_rca': {'n_components': 320},
            'dt': {'cutoff': 0.95}
        }
    else:
        raise ValueError('Invalid data set name!')

```

To reproduce the results in the report, please run - 
```commandline
# python clustering_on_reduced_dimension.py [DATASET_NAME] [CLUSTERING_ALGORITHM] 
# valid DATASET_NAME can be "campaign_marketing" and "university_recommendation"
# valid CLUSTERING_ALGORITHM can be "em" and "kmeans"
# for example:
python clustering_on_reduced_dimension.py campaign_marketing em 
```

To visualize the results and reproduce the plots in the report, please run - 

```commandline
# for dataset: campaign_marketing
python visualize_cluster_on_reduced_dimension.py campaign_marketing


# for dataset: university_recommendation
python visualize_cluster_on_reduced_dimension.py university_recommendation
```

#### Build ANN on Transformed/clustered Data Space 
As mentioned in the report, two network structures are adopted to analyze the ANN performance.
Feel free to change the parameters to explore structures. 
```python
# in dnn.py
PARAMS = {'tiny': {'hidden_units': [8]},
          'medium': {'hidden_units': [16, 8, 4]}}
```
To reproduce the results in the report, please run - 
```commandline
# for dataset: campaign_marketing
python dnn.py campaign_marketing


# for dataset: university_recommendation
python dnn.py university_recommendation
```

To visualize the results and reproduce the plots in the report, please run - 

```commandline
# for dataset: campaign_marketing
python visualize_dnn.py campaign_marketing


# for dataset: university_recommendation
python visualize_dnn.py university_recommendation
```


  

