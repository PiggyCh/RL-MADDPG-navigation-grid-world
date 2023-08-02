import sys
from pathlib import Path
sys.path.append(Path(__file__).parent.parent.resolve().as_posix())

import numpy as np
import sys
import torch
from torch.distributions.categorical import Categorical
import random
from arguments import Args
# process the inputs

env_params = Args.env_params
train_params = Args.train_params
dim_observation = env_params.dim_observation
n_agents = env_params.n_agents
dim_action = env_params.dim_action
dim_hand = env_params.dim_hand
noise_eps = train_params.noise_eps
action_max = env_params.action_max
random_eps = train_params.random_eps
clip_obs = Args.clip_obs
clip_range = Args.clip_range

@torch.no_grad()
def select_action(actors, obs, explore):
    actions = np.ones([n_agents, dim_action])
    # TODO: check noise_eps \ action_max
    # for i in range(n_agents):
    sb_norm = np.array(obs)
    action = actors(num_to_tensor(sb_norm)).cpu().numpy().squeeze()
        # add the gaussian
    if explore:
        if random.random() < 0.1:
            action = np.random.uniform(-1, 1, action.shape)
        action += noise_eps * action_max * np.random.randn(*action.shape)
        action = np.clip(action, -action_max, action_max)
            # random actions...
            # random_actions = np.random.uniform(low=-action_max, high=action_max,
            #                                 size=dim_action)
            # # choose if use the random actions
            # action += np.random.binomial(1, random_eps, 1)[0] * (random_actions - action)

    actions = np.clip(action, 0, 1)
    return actions

def num_to_tensor(inputs, device = 'cpu'):
    # inputs_tensor = th.tensor(inputs, dtype=th.float32).unsqueeze(0)  # 会在第0维增加一个维度
    inputs_tensor = torch.tensor(inputs, dtype=torch.float32).to(device)
    return inputs_tensor

def compute_reward(achieved_goal, goal, sample=False):
    # Compute distance between goal and the achieved goal.
    d = np.linalg.norm(achieved_goal - goal, axis=-1)
    if Args.task_params.reward_type == 'sparse':  # 稀疏奖励
        return -(d > Args.task_params.distance_threshold).astype(np.float32)  # 如果达到目标，返回0，没达到目标，返回-1
    else:
        return -d