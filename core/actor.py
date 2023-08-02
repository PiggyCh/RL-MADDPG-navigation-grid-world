
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

@torch.no_grad()
def actor_worker(
    data_queue,
    actor_queue,
    actor_index,
    logger,
    origin_obstacle_states
):
    try:
        logger.info(f"Actor {actor_index} started.")
        # init env
        env = Gridworld(obstacles=origin_obstacle_states)
        store_item = ['obs',  'next_obs', 'acts', 'r']
        policy = actor(env_params)
        init_flag = False
        rolltime_count = 0
        # sampling ..
        while True:
            # update model params periodly
            if not actor_queue.empty():
                data = actor_queue.get()
                policy.load_state_dict(data['actor_dict'])
                init_flag = True
            # first time initialization
            elif not init_flag:
                time.sleep(5)
                continue
            mb_store_dict = {item : [] for item in store_item}
            rolltime_count += 1
            for rollouts_times in range(store_interval):
                ep_store_dict = {item : [] for item in store_item}
                obs = env.reset() # reset the environment
                # start to collect samples
                for t in range(max_timesteps):
                    actions = select_action(policy, obs, explore = True)  # 输入的是numpy
                    next_obs, reward, done, info = env.step(actions)
                    save_fig = info[0]
                    save_fig_path = f'results_png/demo_{rolltime_count}_{rollouts_times}.png' if save_fig else None
                    env.render(reward, done, save_fig_path)
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
                for key in store_item:
                    mb_store_dict[key].append(deepcopy(ep_store_dict[key]))
            # convert them into arrays
                store_data = [np.array(val) for key, val in mb_store_dict.items()]
                # send data to data_queue
                data_queue.put(store_data, block = True)
            # real_size = self.buffer.check_real_cur_size()
            logger.info(f'actor {actor_index} send data, current data_queue size is {store_interval * data_queue.qsize()}')
    except KeyboardInterrupt:
        logger.critical(f"interrupt")
    except Exception as e:
        logger.error(f"Exception in worker process {actor_index}")
        traceback.print_exc()
        raise e

if __name__ == "__main__":
        # test actor
    from core import actor

    # #! /usr/bin/env python
    import random
    import torch
    import time
    import torch.multiprocessing as mp
    import numpy as np 

    from arguments import Args as args
    from core.logger import Logger
    from core.actor import actor_worker

    import os

    # set logging level 
    logger = Logger(logger="dual_arm_multiprocess")
    train_params = args.train_params
    env_params = args.env_params
    actor_num = train_params.actor_num
    model_path = os.path.join(train_params.save_dir, train_params.env_name)
    if not os.path.exists(train_params.save_dir):
        os.mkdir(train_params.save_dir)
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    # starting multiprocess
    ctx = mp.get_context("spawn") # using shared cuda tensor should use 'spawn'
    # queue to transport data
    data_queue = ctx.Queue()
    actor_queues = [ctx.Queue() for _ in range(1)]
    actor_processes = []
    for i in range(1):
        actor = ctx.Process(
            target = actor_worker,
            args = (
                data_queue,
                actor_queues[i],
                i,
                logger,
            )
        )
        logger.info(f"Starting actor:{i} process...")
        actor.start()
        actor_processes.append(actor)
        time.sleep(1)
