# Introductions to Run Source Code

## Project 3
### Setup the Environment
Follow the steps to set up your environment:
* Make sure python 3 is installed.
* Create and enter the virtual environment using following commend in the bash. 
Make sure the `virtualenv` is under python3 :
```commandline
virtualenv RL2
source RL2/bin/activate
```

* Go inside the `7642Spring2020xzhou94/project3/` folder to install the required packages.
```commandline
cd 7642Spring2020xzhou94/project3/
pip install -r requirements_project3.txt
``` 



### Run Code
Make sure your are in the `RL2` virtual environment and your environment is well set up
before running the code. Since each training needs around 10 mins and upto 30 mins to converge
into a statble policy. I intentionally save all the errors and policies down to the local so that 
they can be easy to reuse for plots and analyses. 
Please create following folders under `7642Spring2020xzhou94/project3/`.
```commandline
mkdir data_store
mkdir plots
```

#### Test Soccer Environment
The soccer environment class is in `soccer_env.py`, with a test case presented. 
Run the following commandline to render an example episode and see the transition.

```commandline
python soccer_env.py
```

Be careful about how stick, E, S, W and N is coded in the environment.
You can add more test cases in this script.
```python
action_dict = {'stick': [0, 0],
               'E': [0, 1],
               'S': [1, 0],
               'W': [0, -1],
               'N': [-1, 0]}
``` 


#### Train the Alogrithms
Run the `main_q_learning.py`,  `main_friend_q.py`,  `main_foe_q.py` and  `main_ce_q.py` scripts to train
Q learning, friend-Q, foe-Q and uCE-Q algorithms for the soccer game.
 
The errors though the training will be stored in the `data_store/` folder for analysis.
For foe-Q and uCE-Q, we also have the converged policy vector scored in `data_store/`.
For friend-Q, the final policy of state s will be printed out in the terminal - it is just two numbers. 
```commandline
python main_q_learning.py
python main_friend_q.py 
python main_foe_q.py
python main_ce_q.py
```
The individual algorithm plots are also stored in the ``plots/`` folder for debugging and 
hyperparameter tuning purpose.


#### Reproduce the Plots in My Paper
After training all four models, run the following commandline to reproduce the 4 x 4 figure in my paper.
```commandline
python plot.py
```