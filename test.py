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
from core.evaluator import evaluate_worker
from core.learner import learn
import os

# set logging level 
logger = Logger(logger="dual_arm_multiprocess")
)

def train():
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
    evalue_queue = ctx.Queue()
    actor_queues = [ctx.Queue() for _ in range(actor_num)]

    actor_processes = []
    for i in range(actor_num):
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
