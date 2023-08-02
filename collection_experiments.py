
import sys
from pathlib import Path
sys.path.append(Path(__file__).parent.parent.resolve().as_posix())

from core.util import select_action
from core.model import actor
from Env.env import Gridworld
from arguments import Args

import torch
import time
import traceback
import numpy as np 
from copy import deepcopy

env_params = Args.env_params
train_params = Args.train_params
max_timesteps = env_params.max_timesteps
store_interval = train_params.store_interval
n_agents = env_params.n_agents

def actor_worker():
        # init env
        env = Gridworld()
        store_item = ['obs',  'next_obs', 'acts', 'r']
        policy = actor(env_params)
        init_flag = False
        rolltime_count = 0
        mb_store_dict = {item : [] for item in store_item}
        rolltime_count += 1
        for rollouts_times in range(store_interval):
            ep_store_dict = {item : [] for item in store_item}
            obs = env.reset() # reset the environment
            # start to collect samples
            for t in range(max_timesteps):
                actions = select_action(policy, obs, explore = True)  # 输入的是numpy
                next_obs, reward, done, info = env.step(actions)
                is_done = info[0]
                store_data = {
                    'obs' : obs, 
                    'next_obs': next_obs if t != max_timesteps - 1 else obs,
                    'acts' : actions,
                    'r': reward
                }
                # append rollouts
                for key, val in store_data.items():
                    ep_store_dict[key].append(val.copy())
                obs = next_obs
            if is_done:
                for key in store_item:
                    mb_store_dict[key].append(deepcopy(ep_store_dict[key]))

