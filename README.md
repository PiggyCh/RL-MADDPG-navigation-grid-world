# RL-MADDPG-navigation-gird-world
An implementation of MADDPG on grid world navigation
use multiprocess, and there is worker for sampling, learner for learning policy, and evaluator for evaluate policy at a fixed interval.

```
├── Env          # 
|   ├── env.py     # base env gird world
├── Core          training core# 
|   ├── HER.py     # Hindsight experience replay, actually not use
|   ├── actor.py     # actor node for sampling data
|   ├── evaluator.py     # evaluate policy node
|   ├── learner.py     # learning process node
|   ├── logger.py     # record necessary info
|   ├── model.py     # define NN model
|   ├── normalizer.py     # normalize sampling data
|   ├── util.py     # some utils
├── arguments                       #train arguments
├── collection_experiments.py       # simple visualize, may use for collecting samples, but not use till now.
├── origin_obstacle_states.txt      # make obstacle same at all evironment, for simplifing problem
├── plot.py                         #plot
├── rollout_test.py                 # rollout saved policy
├── train.py                         #training entrance
```

```python
python train.py # modify args in arguments.py
```
<div style="display: flex; justify-content: center;">
<img src="picture\DMADDPG.png" alt="structure" width="60%" />
</div>
The results:
<div style="display: flex; justify-content: center;">
<img src="picture\plot.png" alt="plot data" width="50%" />
</div>
<div style="display: flex; justify-content: center;">
<img src="picture\result_1.jpg" alt="results after average moving" width="50%" />
</div>
<div style="display: flex; justify-content: center;">
<img src="picture\result_2.gif" alt="visualize" width="60%" />
</div>
