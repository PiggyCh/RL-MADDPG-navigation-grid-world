#! /usr/bin/env python
import sys
from pathlib import Path
sys.path.append(Path(__file__).parent.parent.resolve().as_posix())

import time
import os
import csv
from matplotlib import pyplot as plt
import pygame
from core.util import select_action
from core.model import actor
from Env.env import Gridworld

def initialize_csv(path):
    file_name = os.path.join(path, 'training_data.csv')
    keys = ['step' , 'actor_loss', 'critic_loss', 'rewards']
    with open(file_name, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(keys)
    return file_name

import imageio
if not os.path.exists('results_png'):
    os.mkdir('results_png/')
if not os.path.exists('results_eval_png'):
    os.mkdir('results_eval_png/')

def evaluate_worker(
        train_params,
        env_params,
        plot_path,
        evalue_time,
        evalue_queue,
        logger,
        origin_obstacle_states
    ):
    csv_file_name = initialize_csv(plot_path)
    env = Gridworld(obstacles=origin_obstacle_states)
    actors = actor(env_params)
    total_evalue_time = 0 
    while True:
        if not evalue_queue.empty():
            total_evalue_time += 1
            data = evalue_queue.get()
            evaluate_step = data['step']
            actors.load_state_dict(data['actor_dict'])
            actors.eval()
            total_rewards = []
            max_timesteps = env_params.max_timesteps
            for i in range(evalue_time):
                reward_tmp = []
                obs = env.reset() # reset the environment

                # start to do the demo
                for t in range(max_timesteps):
                    actions = select_action(actors, obs, explore = False)  # 输入的是numpy
                    # put actions into the environment
                    observation_new, reward, dones, info = env.step(actions)
                    # print(f'reward {reward} actions {actions} , dones: {done}')
                    save_fig_path = f'results_eval_png/demo_{total_evalue_time}_{i}.png' if t == max_timesteps - 1 else None
                    env.render(reward, dones, save_fig_path)
                    obs = observation_new
                    reward_tmp.append(sum(reward))
                    if dones[0]:
                        break
                # save frame
                total_rewards.append(sum(reward_tmp))

            logger.info(f' evaluate_step : {evaluate_step} success rate:{sum(reward_tmp)/evalue_time}')
            plot(
                train_params.env_name,
                f'{plot_path}/plot.png', 
                csv_file_name, 
                {
                    'step' : evaluate_step,
                    'actor_loss' : data['actor_loss'],
                    'critic_loss': data['critic_loss'],
                    'rewards': sum(reward_tmp)/evalue_time
                }
            )
        else:
            time.sleep(30)

def plot(test_env_name, plot_path, csv_file_name, data):
    total_data = {key : [] for key in data.keys()}
    # write
    with open(csv_file_name, mode = 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([value for value in data.values()])
    # read 
    with open(csv_file_name, mode = 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in total_data.keys():
                total_data[key].append(float(row[key]))
    total_data = {key : val for key,val in total_data.items()}
    N_COLUMN = 3
    fig, axes = plt.subplots(nrows = 1, ncols = N_COLUMN, figsize=(18,6))
    fig.suptitle(test_env_name, fontsize=10)

    for i, key in enumerate(total_data.keys()):
        if key == 'step':
            continue
        ax = axes[i-1]
        ax.ticklabel_format(style='sci', scilimits=(-1,2), axis='x')
        ax.set_title(key)
        ax.plot(total_data["step"], total_data[key])
    plt.savefig(plot_path)
    plt.close()


#  from random import random
#  import numpy as np
#  from arguments import Args
# time = 0
# def generate_data(): 
#     global time
#     time += 1
#     return  {
#         'step' :2000*time,
#         'actor_loss' : 0.5 + random(),
#         'critic_loss': 0.6 + random(),
#         'success_rate': 0.8 + random()
#     }
# test_env_name = 'armrobot_push_ seed125_10_31_21'
# plot_path = os.path.join(Args.train_params.save_dir, test_env_name)
# csv_file_name = initialize_csv(plot_path, generate_data().keys())  
# for _ in range(1000):
#     plot(
#         test_env_name,
#         f'{plot_path}/plot.png', 
#         csv_file_name, 
#         generate_data()
#     )
