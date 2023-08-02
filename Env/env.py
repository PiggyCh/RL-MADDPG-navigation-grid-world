import numpy as np
import matplotlib.pyplot as plt
import pygame
import random
import gym
import math
import copy
import sys 
from easydict import EasyDict as edict

env_params = edict({    
    'grid_size' : 20,
    'n_agents' :  3,
    'observation_dim' : 35, 
    'action_dim' : 5,
    'clip_obs' : False,
    'max_timesteps' : 300,
    })


class Gridworld(gym.Env):
    def __init__(self, agent_num = 3, obstacles = None):
        # Initialize pygame
        pygame.init()
        self.agent_num = agent_num
        self.seed = 10
        self.save_fig_time = 0
        # Set up window and font
        self.font = pygame.font.SysFont(None, 30)
        self.window_size = (400, 400)
        self.window = pygame.display.set_mode(self.window_size)
        pygame.display.set_caption('Gridworld')
        self.save_max_num = 10
        
        # Initialize gridworld
        self.grid_size = 20
        self.num_actions = 5
        self.action_mapping = {0:[0,0],1:[-1,0],2:[1,0],3:[0,-1],4:[0,1]}
        self.num_states = self.grid_size * self.grid_size
        
        self.origin_obstacle_states = []
        self.origin_stable_obstacle_states = copy.deepcopy(obstacles)
        self.goal_state = []
        # init obstacle pos
        # for _ in range(60):
        #     x = random.randint(1,self.grid_size-1 )
        #     y = random.randint(1,self.grid_size-1 )
        #     if [x,y] not in self.origin_obstacle_states and [x,y] not in self.goal_state:
        #         self.origin_stable_obstacle_states.append([x,y])
        # init agent pos
        origin_pos_tmp = agent_num
        self.origin_current_state = []
        while origin_pos_tmp > 0:
            x = random.randint(1, self.grid_size - 1)
            y = random.randint(1, self.grid_size - 1)
            if [x, y] not in self.origin_obstacle_states and [x, y] not in self.goal_state and [x,y] not in self.origin_current_state and [x,y] not in self.origin_stable_obstacle_states:
                self.origin_current_state.append([x,y])
                origin_pos_tmp -= 1
        # init goal pos\
        self.obstacle_movement_prob = 0.05
        self.max_step = 200

        # define for RL
        cp_obs_space = 3 * self.grid_size * self.grid_size + 2 * len(self.goal_state) + 4 + 1 + 2 # map 3 * 20 * 20 + 15
        # self.observation_space = gym.spaces.Box(low = -1, high=1, shape = (4, self.grid_size, self.grid_size,))
        self.observation_space = gym.spaces.Box(low = -1, high=1, shape = (cp_obs_space,))
        self.action_space = [
           gym.spaces.Box(low = -1, high=1, shape = (1,)) for _ in range(self.agent_num) # 0,1 for move, 2,3 for 20 load\unload, 4,5 for 40 load\unload
        ]
    
        self.reset()

    def sample_goals(self):
        self.goal_state = []
        goal_pos_tmp = self.agent_num
        while goal_pos_tmp > 0:
            x = random.randint(1, self.grid_size - 1)
            y = random.randint(1, self.grid_size - 1)
            if [x, y] not in self.origin_obstacle_states and [x, y] not in self.goal_state and [x,y] not in self.origin_current_state and [x,y] not in self.origin_stable_obstacle_states:
                self.goal_state.append([x,y])
                goal_pos_tmp -= 1
    
    def get_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def reset(self):
        self.trajectory = [[] for _ in range(self.agent_num)]
        self.stable_obstacle_states = copy.deepcopy(self.origin_stable_obstacle_states)
        self.current_state = copy.deepcopy(self.origin_current_state)
        self.obstacle_states =  copy.deepcopy(self.origin_obstacle_states)
        self.sample_goals()
        self.cur_step = 0
        self.get_goal = [0] * self.agent_num
        self.last_dis = [0] * self.agent_num
        for i in range(self.agent_num):
            for j in range(self.agent_num):
                self.last_dis[i] = min(self.last_dis[i], self.get_distance(self.current_state[i], self.goal_state[j]))
        state = [self.get_state(i) for i in range(self.agent_num)]
        return state
    
    def get_state(self, i):
        total_obs = [] 
        total_obs.append(self.cur_step / self.max_step)
        # agent pos
        my_x, my_y = self.current_state[i]
        total_obs.append(my_x/self.grid_size)
        total_obs.append(my_y/self.grid_size)
        for j in range(self.agent_num):
            x, y = self.current_state[j]
            total_obs.append(x/self.grid_size)
            total_obs.append(y/self.grid_size)
            goal_x, goal_y = self.goal_state[j]
            total_obs.append(goal_x/self.grid_size)
            total_obs.append(goal_y/self.grid_size)
            total_obs.append((my_x - goal_x) / self.grid_size)
            total_obs.append((my_y - goal_y) / self.grid_size)
            total_obs.append((my_x - x) / self.grid_size)
            total_obs.append((my_y - y) / self.grid_size)
        
        # get available action
        total_obs.extend(self.get_availabel_action(i))
        agent_id = [0, 0, 0]
        agent_id[i] = 1
        total_obs.extend(agent_id)
        # is in goal
        is_in_goal = 1 if self.current_state[i] in self.goal_state else 0
        total_obs.append(is_in_goal)
        return total_obs
    
    def get_availabel_action(self, agent_id):
        direction = list(self.action_mapping.values())
        available_action = [1] * 5
        for i, direc in enumerate(direction):
            new_x, new_y = self.current_state[agent_id][0] + direc[0], self.current_state[agent_id][1] + direc[1]
            if [new_x, new_y] in self.stable_obstacle_states or ([new_x, new_y] in self.stable_obstacle_states) or new_x < 0 or new_y < 0 or new_x >= self.grid_size or new_y >= self.grid_size:
                available_action[i] = 0
            for j in range(self.agent_num):
                if new_x == self.current_state[j][0] and new_y == self.current_state[j][1]:
                    available_action[i] = 0
        available_action[0] = 1
        return available_action
    
    def savefig(self, name):
        self.save_fig_time  += 1
        if self.save_max_num > self.save_fig_time:
            pygame.image.save(self.window, f"path_Saving_{name}_{self.save_fig_time}.png")

    def parse_action(self, actions):

        new_actions = np.argmax(actions, axis= -1)

        return new_actions

    def step(self, actions):
        actions = self.parse_action(actions)
        self.cur_step += 1
        for action in actions:
            if action < 0 or action >= self.num_actions:
                raise Exception('Invalid action: {}'.format(action))
        assert len(actions) == self.agent_num, f'actions length is {len(actions)}, agent_num {self.agent_num}'
        rewards = [-0.005] * self.agent_num
        for i, action in enumerate(actions):
            if self.current_state[i] in self.goal_state:
                rewards[i] += 0.005
                continue
            valid_actions = self.get_availabel_action(i)
            if valid_actions[action] == 0:
                # do nothing but get a negative reward
                rewards[i] -= 0.005
            else:
                next_state = [self.current_state[i][0] + self.action_mapping[action][0], self.current_state[i][1] + self.action_mapping[action][1]]
                shorted_dis = min(self.get_distance(next_state, self.goal_state[j]) for j in range(self.agent_num))
                if shorted_dis < self.last_dis[i]:
                    rewards[i] += 0.01
                    self.last_dis[i] = shorted_dis
                self.current_state[i] = next_state
                self.trajectory[i].append(self.current_state[i])
                if self.current_state[i] in self.goal_state:
                    idx = self.goal_state.index(self.current_state[i])
                    if self.get_goal[idx] == 0:
                        rewards[i] += 1
                        self.get_goal[idx] = 1
        done, total_r = self.get_is_done()
        rewards = [r+total_r for r in rewards]
        info = [] if total_r != 1 else [True]
        sub_agent_obs = []
        sub_agent_reward = []
        sub_agent_done = []
        sub_agent_info = []
        for i in range(self.agent_num):
            sub_agent_obs.append(self.get_state(i))
            sub_agent_reward.append(rewards[i])
            sub_agent_done.append(done)
            sub_agent_info.append(info)

        return sub_agent_obs, sub_agent_reward, sub_agent_done, sub_agent_info
        
    def get_is_done(self):
            if self.cur_step >= self.max_step:
                return True, 0
            if sum(self.get_goal) != self.agent_num:
                return False, 0
            return True, 1

    def reward_func(self):
        rewards = [0] * self.agent_num
        return rewards
    
    def render(self, reward, done, save_path_name = None):
        # Clear window
        self.window.fill((200, 200, 200))
        row_size = self.window_size[0] / self.grid_size
        col_size = self.window_size[1] / self.grid_size
        
        # Draw grid lines
        for i in range(self.grid_size+1):
            pygame.draw.line(self.window, (0, 0, 0), (0, i*col_size), (self.window_size[0], i*col_size), 1)
            pygame.draw.line(self.window, (0, 0, 0), (i*row_size, 0), (i*row_size, self.window_size[1]), 1)

        # Draw obstacles
        for obstacle_state in self.origin_stable_obstacle_states:
            color = (0, 0, 0)
            # if obstacle_state in self.stable_obstacle_states:
            #     color = (0, 0, 0)
            pygame.draw.rect(self.window, color, (obstacle_state[1]*row_size, obstacle_state[0]*col_size, row_size, col_size))

        # Draw goal state
        for goal in self.goal_state:
            color = (255, 0, 0)
            # if self.get_goal[self.goal_state.index(goal)] == 0:
            #     color = (0, 255, 0)
            pygame.draw.rect(self.window, color, (goal[1]*row_size, goal[0]*col_size, row_size, col_size))

        # Draw agent
        for i in range(self.agent_num):
            x, y = self.current_state[i][0], self.current_state[i][1]
            pygame.draw.rect(self.window, (1, 25, 230), (y*row_size, x*col_size, row_size, col_size))

        # Draw trajectory
        for agent_traj in self.trajectory:
            for point in agent_traj:
                pygame.draw.circle(self.window, (111, 25, 230), ((0.5+point[1])*row_size, (0.5+point[0])*col_size), 5, width=1)

        # Draw reward and done status
        reward_text = self.font.render('Reward: {}'.format(reward), True, (0, 0, 0))
        done_text = self.font.render('Done: {}'.format(done), True, (0, 0, 0))
        self.window.blit(reward_text, (10, self.window_size[1]-40))
        self.window.blit(done_text, (10, self.window_size[1]-70))

        # Update display

        pygame.display.update()
        if save_path_name is not None:
            pygame.image.save(self.window, save_path_name)

        # Check for quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

    def euclidean_distance(self, point1, point2):
        return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])

if __name__ == "__main__":
    import time
    def test(env, num_steps=1000):
        for i in range(10):
            total_reward = 0
            env.reset()
            env.render(0, False)
            for _ in range(num_steps):
                action = [np.random.randint(0, env.num_actions - 1 ) for i in range(env.agent_num)]
                next_state, reward, done, _ = env.step(action)
                env.render(sum(reward), done[0])
                total_reward += sum(reward) 
                time.sleep(0.1)
                if done[0]:
                    break
            print('Total reward: {}'.format(total_reward))
    env = Gridworld()
    test(env)