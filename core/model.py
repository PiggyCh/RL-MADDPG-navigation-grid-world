import torch as th
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

agent_num = 3 
class critic(nn.Module):
    def __init__(self, env_params):
        super(critic, self).__init__()
        self.max_action = env_params.action_max
        self.FC1 = nn.Linear(36 * agent_num + 15, 256)   # 48+6+6+3
        self.FC2 = nn.Linear(256, 256)
        self.q_out = nn.Linear(256, 1)
        self.RELU = nn.ReLU()

    def forward(self, obs, acts):  # 前向传播  acts 6   hand 6
        combined = th.cat([obs, acts], dim=-1)  # 将各个agent的观察和动作联合到一起
        result = self.RELU(self.FC1(combined))  # relu为激活函数 if输入大于0，直接返回作为输入值；else 是0或更小，返回值0。
        result = self.RELU(self.FC2(result))
        q_value = self.q_out(result)
        return q_value

class actor(nn.Module):
    def __init__(self, env_params):
        super(actor, self).__init__()
        self.max_action = env_params.action_max
        self.FC1 = nn.Linear(36, 256)  # 24+3
        self.FC2 = nn.Linear(256, 256)   # FC为full_connected ，即初始化一个全连接网络
        self.action_out = nn.Linear(256, env_params.dim_action)
        self.RELU = nn.ReLU()
        self.Tanh = nn.Tanh()

    def forward(self, obs_and_g):
        result = self.RELU(self.FC1(obs_and_g))
        result = self.RELU(self.FC2(result))
        actions = F.softmax(self.action_out(result), dim=-1)  # hand_logits 为末端状态选择的概率 3维
        return actions

class Net():
    def __init__(self, env_params, device = 'cpu'):
        self.device = device
        self.env_params = env_params
        self.n_agents = env_params.n_agents
        # self.disc = Discriminator(args.env_params)  # if imitation learning
        self.actor = actor(env_params).to(device) 
        self.critic = critic(env_params).to(device)
        self.actors_target = deepcopy(self.actor)
        self.critics_target = deepcopy(self.critic)
        # load the weights into the target networks 可以将预训练的参数权重加载到新的模型之中
        self.actors_target.load_state_dict(self.actor.state_dict())
        self.critics_target.load_state_dict(self.critic.state_dict())

    def update(self, model): 
        self.actor.load_state_dict(model.actor.state_dict())
