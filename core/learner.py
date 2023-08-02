import torch
import time
import numpy as np
from torch.optim import Adam
from core.buffer import replay_buffer
from copy import deepcopy
from arguments import Args
from core.model import Net

env_params = Args.env_params
train_params = Args.train_params

n_agents = env_params.n_agents
dim_hand = env_params.dim_hand
device =  train_params.device
batch_size = train_params.batch_size
learner_step = train_params.learner_step
clip_obs = train_params.clip_obs
gamma = train_params.gamma
polyak = train_params.polyak
evalue_interval = train_params.evalue_interval

preproc = lambda norm_func, tensor_data : norm_func(torch.clamp(tensor_data, -clip_obs, clip_obs))

def store_buffer(buffer, data_queue):
    for _ in range(data_queue.qsize()):
        buffer.push(data_queue.get(block = True))

# # for Loss calculate
# def get_action(actors_model, obs):
#     batch_size = obs.shape[0]
#     act = actors_model[i](input_tensor)
#     return acts.view(batch_size, -1)

def get_value(critics, obs, acts_tensor):
    input_tensor = obs
    q_value = critics(input_tensor, acts_tensor)
    return q_value

def update_network(
        model,
        transitions,
        actor_optimizer,
        critic_optimizer
    ):
    # pre-process the observation and goal
    obs, obs_next = transitions['obs'], transitions['next_obs']
    acts_tensor, r_tensor = transitions['acts'], transitions['reward']
    batch = obs.shape[0]
    

    with torch.no_grad():
        
        # calculate the target Q value function
        obs = obs.reshape(batch, n_agents,  -1)
        obs_next = obs_next.reshape(batch, n_agents,  -1)
        acts_i = acts_tensor.reshape(batch, n_agents, -1)
        r_tensor_i = r_tensor.reshape(batch, n_agents, -1)
    
        acts_next_tensor = model.actors_target(obs_next).reshape(batch, -1).unsqueeze(1).repeat(1, 3, 1)
        q_next_value = get_value(model.critics_target, obs_next.reshape(batch, -1).unsqueeze(1).repeat(1, 3, 1), acts_next_tensor)
        target_q_value = r_tensor_i + gamma * q_next_value
        # clip the q value
        # clip_return = 1 / (1 - gamma)
        # target_q_value = torch.clamp(target_q_value, -clip_return, 0)
    # the q loss
    real_q_value = get_value(model.critic, obs.reshape(batch, -1).unsqueeze(1).repeat(1, 3, 1), acts_i.reshape(batch, -1).unsqueeze(1).repeat(1, 3, 1))
    critic_loss = (target_q_value - real_q_value).pow(2).mean()
        # update the critic_network
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    # the actor loss
    acts_real_tensor = model.actor(obs)
    actor_loss = -get_value(model.critic, obs.reshape(batch, -1).unsqueeze(1).repeat(1, 3, 1), acts_real_tensor.reshape(batch, -1).unsqueeze(1).repeat(1, 3, 1)).mean()
    # actor_loss += self.args.action_l2 * (acts_real_tensor / self.env_params['action_max']).pow(2).mean()

    # start to update the network
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()



    return actor_loss, critic_loss

def learn(model_path, data_queue, evalue_queue, actor_queues, logger):
    # initialize function here
    learner_model = Net(env_params, device)
    buffer = replay_buffer(env_params, train_params, logger)
    actor_optimizer = Adam(learner_model.actor.parameters(), lr= train_params.lr_actor)
    critic_optimizer = Adam(learner_model.critic.parameters(), lr= train_params.lr_critic)
    Actor_loss, Critic_loss = 0, 0
    savetime = 0
    for queue in actor_queues:
        queue.put({
            'actor_dict' : deepcopy(learner_model.actor).cpu().state_dict(),
            })
    # waiting buffer data before training
    while buffer.current_size < batch_size:
        store_buffer(buffer, data_queue)
        logger.info(f'wating for samples... buffer current size {buffer.current_size}')
        time.sleep(5)
    
    for step in range(1, train_params.learner_step):
        # self._update_network_gail() # if need Imitation Learning  
                # sample the episodes
        transitions = buffer.sample(batch_size)
        training_data = update_network(
            learner_model,
            transitions,
            actor_optimizer,
            critic_optimizer
        )
        Actor_loss += training_data[0]
        Critic_loss += training_data[1]
        # soft update
        def soft_update_target_network(target, source):
            for target_param, source_param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_((1 - polyak) * source_param.data + polyak * target_param.data)

        if step % train_params.update_tar_interval == 0:
            logger.info(f'cur step: {step}')
            soft_update_target_network(learner_model.actors_target, learner_model.actor)
            soft_update_target_network(learner_model.critics_target, learner_model.critic)
        # start to do the evaluation
        if step % evalue_interval == 0:
            store_buffer(buffer, data_queue)
            Actor_loss /= evalue_interval
            Critic_loss /= evalue_interval
            logger.info(f'cur step: {step}, actor loss:{Actor_loss:.4f}, critic loss:{Critic_loss:.4f}')
            model_params = {
                'actor_dict' : deepcopy(learner_model.actor).cpu().state_dict(),
            }
            for queue in actor_queues:
                queue.put(model_params)
            evalue_params = deepcopy(model_params)
            evalue_params.update(
                { 
                    'step' : step,
                    'actor_loss' : Actor_loss.item() ,
                    'critic_loss' : Critic_loss.item()
                }
            )
            # synchronize the normalizer and actor_worker model not too frequency
            evalue_queue.put(evalue_params)
            # save model
            torch.save([
                    learner_model.actor.state_dict(), 
                    learner_model.critic, ],#, self.disc.state_dict()],
                    model_path + '/' + str(train_params.seed) + '_' +  str(savetime) + '_model.pt')
            savetime += 1
            Actor_loss, Critic_loss = 0, 0
        # if epoch >= 80:
        #     self.Is_train_discrim = False
        #     self.theta *= 0.9  # annealing
