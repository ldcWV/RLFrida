from constants import CANVAS_SIZE
from src.ddpg.replay_buffer import ReplayBuffer
from src.ddpg.actor import ResNet
from src.ddpg.critic import ResNet_wobn
from src.ddpg.utils import hard_update, soft_update
import torch

class DDPG:
    def __init__(self, env):
        self.replay_buffer = ReplayBuffer(50000)

        self.env = env

        self.coordconv = torch.zeros((1, 2, CANVAS_SIZE, CANVAS_SIZE))
        for i in range(CANVAS_SIZE):
            for j in range(CANVAS_SIZE):
                self.coordconv[0, 0, i, j] = i / CANVAS_SIZE
                self.coordconv[0, 1, i, j] = j / CANVAS_SIZE

        self.actor_net = ResNet(9, 18, self.env.action_space) # target, canvas, stepnum, coordconv 3 + 3 + 1 + 2
        self.actor_target_net = ResNet(9, 18, self.env.action_space)
        self.critic_net = ResNet_wobn(9, 18, 1)
        self.critic_target_net = ResNet_wobn(9, 18, 1)

        self.device = 'cpu'

        hard_update(self.actor_net, self.actor_target_net)
        hard_update(self.critic_net, self.critic_target_net)

    def add_to_replay_buffer(self, obs, action, reward, next_obs, done, goal):
        '''Store the transition into the replay buffer'''
        obs = torch.tensor(obs, device='cpu')
        action = torch.tensor(action, device='cpu')
        reward = torch.tensor(reward, device='cpu')
        next_obs = torch.tensor(next_obs, device='cpu')
        done = torch.tensor(done, device='cpu')
        goal = torch.tensor(goal, device='cpu')

        self.replay_buffer.add((obs, action, reward, next_obs, done, goal))

    def select_action(self, obs, goal):
        '''Select an action from the policy network'''
        raise NotImplementedError

    def optimize(self):
        '''Updates policy and value network using a batch from the replay buffer'''
        raise NotImplementedError

    def set_device(self, device):
        self.device = device
        self.actor_net.to(device)
        self.actor_target_net.to(device)
        self.critic_net.to(device)
        self.critic_target_net.to(device)
