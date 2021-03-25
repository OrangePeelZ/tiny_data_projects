# Introductions to Run Source Code

## Project 2
### Setup the Environment
Follow the steps to set up your environment:
* Make sure python 3 is installed.
* Create and enter the virtual environment using following commend in the bash. 
Make sure the `virtualenv` is under python3 :
```commandline
virtualenv RL2
source RL2/bin/activate
```

* Go inside the `7642Spring2020xzhou94/project2/` folder to install the required packages.
```commandline
cd 7642Spring2020xzhou94/project2/
pip install -r requirements_project2.txt
``` 
* Install `box2d-py` so that we can call `Lunar Lander` in the gym. 
Make sure you have `swig` installed on your laptop first.
```commandline
# check if you have the swig installed 
swig -version 
# Install swig if you do not have it.
brew install swig@3
```
```commandline
# Build the box2d-py from github.
pip uninstall box2d-py
git clone https://github.com/pybox2d/pybox2d
cd pybox2d/
python setup.py clean
python setup.py build
sudo python setup.py install
```


### Run Code
Make sure your are in the `RL2` virtual environment and your environment is well set up
before running the code. Since each training needs around 15 mins and upto 30 mins. 
I intentionally save all the rewards and models down to the local so that they are 
reuseable for plots and analyses. Please create folders under `7642Spring2020xzhou94/project2/`.
```commandline
mkdir training_rewards
mkdir test_rewards
mkdir models
mkdir plots
```

#### Quick Start
Run the following to build a simple DQN agent for you Lunar Lander. 
It normally takes ~15mins to build. The model and train time reward will be store at 
`models/demo_model.pt` and `training_rewards/demo_training_rewards.pkl`

```commandline
python demo_sucessful_net.py
```

Run the following to generate the reward on the test dataset 
by using the agent we trained above. The test rewards will be saved at 
`test_rewards/demo_test_rewards.pkl`
```commandline
python run_test.py demo
``` 

Run the folling to plot out the rewards of this demonstration.
```commandline
python plot.py demo
```

#### Parameter Tuning
In the scripts, Four parameters are grid searched: learning rate, gamma, decay_rate 
and network structure. Each parameters has three values, which make the total number
of agents trained to be 81. In the script, the `multiprocessing` is used to parallelize the training. Default is to
use 6 processors. You can adjust it based on you own machine.

Training all the 81 agents takes more than 15hrs. Adjust the parameter dictionary (like following) 
in `main.py` based on what you need.
```python
# grid search parameters
EPS_DECAY = [1000, 10000, 50000]
GAMMA = [0.8, 0.99, 0.999]
LR = [0.01, 0.001, 0.0001]
NETWORK = {'simple': (16, 8),
           'medium': (64, 16),
           'complex': (256, 128)}
```

Runing the `main.py` to trained agents for 81 combination of hyperparameters and 
store all the output models and train time rewards to the directory:
`models/` and `training_rewards/` 
```commandline
python main.py
```

Execute `run_test.py` and specify `hyperparam` to generate the test reward on the 
100 test episodes, and save the them at `test_rewards/`. Note that this step is also 
parallelized and the default processor is 9. Check the available number of processors 
on you machine before start it.

```commandline
python run_test.py hyperparam
```

As you have all the train time rewards, test time rewards and DQN agent models right now. 
You will be able to play with it and do any analysis as you want. Execute the `plot.py` 
script and specify `reproduce` will reproduce the plots I used on my paper.

```commandline
python plot.py reproduce
```



  

