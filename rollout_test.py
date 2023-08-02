#! /usr/bin/env python
import os
import torch as th
from core.model import actor
from Env.env import Gridworld 
from core.util import select_action
from arguments import Args
import sys

#加载训练好的模型 数据
model_path = "saved_models/grid_world_seed125_5_14_14_39/125_4320_model.pt"

origin_obstacle_states = [[1, 11],[1 ,18],
                        [2, 9] , [2,11], [2, 15],[3,2],[3,7],
                        [3,14],[4,15],[4,16],[4,19],
                        [5, 7],[5,14],[6,10],[6,13],[6 ,17],
                        [7 ,6], [7, 9], [7, 12], [7, 14],
                        [8,7],[8,11],[8,18],[9,12],[9,19]
                        ,[10, 2],[10, 8],[10, 9],[10,14],[10, 15], [10,19],
                        [11,2],[11, 16],
                        [12, 5],[12,6],[12,7],[12,11],[12,12],
                        [13, 5],[13, 10],
                        [14,15],[14,19],[15,2],[15,19],[16,5],
                        [17,11],[17,14],[18,10],[18,17],[18,18],
                        [19, 2],[19,8],[19,10],[19,13],[19,14],[19,16]]

def rollout_test(args):
    act, cr, = th.load(model_path, map_location=lambda storage, loc: storage)
    # create the environment+
    env_params = args.env_params
    # get agents
    actors_network = actor(env_params)
    actors_network.load_state_dict(act)
    actors_network.eval()
    env = Gridworld(obstacles=origin_obstacle_states)
    for i in range(args.demo_length):
        reward_tmp = []
        obs = env.reset() # reset the environment
        # start to do the demo
        for t in range(env_params.max_timesteps):
            actions = select_action(actors_network, obs, explore = False)  # 输入的是numpy
            # put actions into the environment
            observation_new, reward, done, info = env.step(actions)
            # print(f'reward {reward} actions {actions} , dones: {done}')
            env.render(reward, done, None)
            obs = observation_new
            reward_tmp.append(sum(reward))


if __name__ == '__main__':
    args = Args()
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    rollout_test(args)

