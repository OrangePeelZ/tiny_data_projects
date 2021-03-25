# Introduction to Run Source Code
The code is also provided at 
https://github.gatech.edu/xzhou94/ML_Fall2020/tree/master/supervised_learning

## Supervised Learning
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
cd supervised_learning/
pip install -r requirements.txt
``` 

### Run Code
Make sure your are in the `gtml` virtual environment and your environment is well set up
before running the code. `campaign_marketing` is a relatively smaller dataset, and the 
complete reproducing take couple of minutes. However, the reproducing of second dataset 
`university_recommendation` will take longer time,  ~15 min on single machine. This 
instruction will lead you to save all the key results of the training in order to 
reproduce the plots and offline metrics in my paper.   
Please create folders under `supervised_learning/`.
```commandline
mkdir EDA
mkdir logdir
mkdir metrics
mkdir plots
```

#### Exploratory Data Analysis
Run the following code will generate the label and feature distributions for selected 
dataset. By observing the distribution, it gives us clues to clean the data and select
the better algorithm and hyperparameters. The EDA plots will be saved in `EDA/` folder. 

```commandline
# for dataset: campaign_marketing
python main.py campaign_marketing EDA


# for dataset: university_recommendation
python main.py university_recommendation EDA
```


#### Model Training
In the experiment, five set of machine learning algorithms are tested. For each algorithm, 
four designed combination of hyperparameters are presented. They are -
* Decision Tree: `min_samples_split`, `max_features`
* Gradient Boosting Tree: `min_samples_split`, `n_estimators`, `learning_rate`
* SVM: `C`, `kernel`
* KNN: `n_neighbors`, `p`
* DNN: `Hidden_units`

Noting that all the parameters are listed in `meta_data.py` file. Feel free to make the 
change and try more combinations of parameters. 

The after training, each learner are evaluated on a common holdout dataset. The results
are stored in `metrics/`, as well as visualization in `plots/`. By comparing the offline 
metrics, we can claim the efficiency for each algorithm on different dataset.

To run a specific algorithm on a specific dataset. You can call 
`python main.py <dataset_name> train_and_evaluation <algorithm> `. For example, 
if you want to run deep neural network on `campaign_marketing` dataset, you can 
type following in the terminal. A validate `dataset_name` can be `campaign_marketing` or `university_recommendation`;
and a validate `algorithm` can be `dt`, `gbt`, `svm`, `knn` or `dnn`.

```commandline
python main.py campaign_marketing train_and_evaluation dnn 
```

Without specifying the `algorithm`, you can reproduce all five sets of algorithm in one
shot. It could take relatively long time for a larger dataset like 
`university_recommendation`. 

```commandline
python main.py campaign_marketing train_and_evaluation
```

### More analysis
For each model, depending on the model type, the code enable analysis of the iteration against
the training and test error. You can run following command 
`python main.py <dataset_name> train_and_evaluation <algorithm> True` to reproduce the plots. The plots 
will be saved in `plots/` with a `futher_analysis_` prefix. 
An example:
```commandline
python main.py campaign_marketing train_and_evaluation dnn True
```



  

