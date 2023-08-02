"""
Here are the params for training
"""
from easydict import EasyDict as edict
import time

class Args:
    time_date = time.localtime()
    date = f'{time_date.tm_mon}_{time_date.tm_mday}_{time_date.tm_hour}_{time_date.tm_min}'
    seed = 125  # 123
    n_agent = 3
    clip_obs = 5
    actor_num = 8
    clip_range = 200
    action_bound = 1
    demo_length = 25  # 20
    Use_GUI = True
    env_params = edict({    
        'n_agents' :  n_agent,
        'dim_observation' : 21, 
        'dim_action' : 5,
        'dim_hand' :  3,
        'dim_achieved_goal' :  3,
        'clip_obs' : clip_obs,
        'dim_goal' :  3,
        'max_timesteps' : 200,
        'action_max' : 1
        })

    train_params = edict({
        # params for multipross
        'learner_step' : int(1e7),
        'update_tar_interval' : 40,
        'evalue_interval' : 240,
        'evalue_time' : 5,  # evaluation num per epoch
        'store_interval': 2,
        'actor_num' : actor_num,
        'date' : date,
        'checkpoint' : None,
        'polyak' : 0.95,  # 软更新率
        'action_l2' : 1, #  actor_loss += self.args.action_l2 * (acts_real_tensor / self.env_params['action_max']).pow(2).mean()
        'noise_eps' : 0.01,  # epsillon 精度
        'random_eps' : 0.3,
        'theta' : 0.1, # GAIL reward weight
        'Is_train_discrim': True,
        'roll_time' : 2,
        'gamma' : 0.98,
        'batch_size' :  256,
        'buffer_size' : 1e6, 
        'device' : 'cpu',
        'lr_actor' : 0.001,
        'lr_critic' : 0.001,
        'lr_disc' : 0.001,
        'clip_obs' : clip_obs,
        'clip_range' : 200,
        'add_demo' : False,
        'save_dir' : 'saved_models/',
        'seed' : seed,
        'env_name' : 'grid_world_' + "seed" +str(seed) + '_' + str(date),
        'demo_name' : 'armrobot_100_push_demo.npz',
        'replay_strategy' : 'future',# 后见经验采样策略
        'replay_k' :  4  # 后见经验采样的参数
    })

    train_params.update(env_params)


