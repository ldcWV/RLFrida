from src.ddpg.replay_buffer import ReplayBuffer
import torch

class DDPG:
    def __init__(self):
        self.replay_buffer = ReplayBuffer(50000)

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

    def set_device(self):
        raise NotImplementedError
