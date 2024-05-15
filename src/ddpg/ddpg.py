from constants import CANVAS_SIZE
from ddpg.replay_buffer import ReplayBuffer
from ddpg.actor import ResNet
from ddpg.critic import ResNet_wobn
from ddpg.utils import hard_update, soft_update
import torch
import torch.optim as optim

class DDPG:
    def __init__(self, env):
        self.tau = 0.001
        self.discount = 0.9
        self.batch_size = 64

        self.replay_buffer = ReplayBuffer(50000)

        self.env = env

        self.device = 'cpu'

        self.coordconv = torch.zeros((1, 2, CANVAS_SIZE, CANVAS_SIZE))
        for i in range(CANVAS_SIZE):
            for j in range(CANVAS_SIZE):
                self.coordconv[0, 0, i, j] = i / CANVAS_SIZE
                self.coordconv[0, 1, i, j] = j / CANVAS_SIZE

        self.actor_net = ResNet(9, 18, self.env.action_dim) # target, canvas, stepnum, coordconv 3 + 3 + 1 + 2
        self.actor_target_net = ResNet(9, 18, self.env.action_dim)
        self.critic_net = ResNet_wobn(9, 18, 1)
        self.critic_target_net = ResNet_wobn(9, 18, 1)

        self.actor_optim = optim.Adam(self.actor_net.parameters(), lr=1e-2)
        self.critic_optim = optim.Adam(self.critic_net.parameters(), lr=1e-2)

        hard_update(self.actor_target_net, self.actor_net)
        hard_update(self.critic_target_net, self.critic_net)

    def add_to_replay_buffer(self, obs, action, reward, next_obs, done, goal):
        '''Store the transition into the replay buffer'''
        obs = obs.cpu()
        action = action.cpu()
        reward = reward.cpu()
        next_obs = next_obs.cpu()
        done = torch.tensor(done, dtype=torch.float)
        goal = goal.cpu()

        self.replay_buffer.add((obs, action, reward, next_obs, done, goal))

    def select_action(self, obs, goal, target=False):
        '''
        Select an action from the policy network
        obs: (batch_size, CANVAS_SIZE, CANVAS_SIZE, 4)
        goal: (batch_size, CANVAS_SIZE, CANVAS_SIZE, 3)
        '''
        not_batched = False
        if obs.ndim == 3:
            obs = obs.unsqueeze(0)
            goal = goal.unsqueeze(0)
            not_batched = True

        batch_size = obs.shape[0]
        obs = obs.permute(0, 3, 1, 2) # (batch_size, 4, CANVAS_SIZE, CANVAS_SIZE)
        goal = goal.permute(0, 3, 1, 2) # (batch_size, 3, CANVAS_SIZE, CANVAS_SIZE)
        inp = torch.cat([obs, goal, self.coordconv.expand(batch_size, 2, CANVAS_SIZE, CANVAS_SIZE)], dim=1)

        if target:
            action = self.actor_target_net(inp)
        else:
            action = self.actor_net(inp)

        if not_batched:
            action = action.squeeze(0)
        return action
    
    def evaluate_state(self, obs, goal, target=False):
        batch_size = obs.shape[0]
        obs = obs.permute(0, 3, 1, 2) # (batch_size, 4, CANVAS_SIZE, CANVAS_SIZE)
        goal = goal.permute(0, 3, 1, 2) # (batch_size, 3, CANVAS_SIZE, CANVAS_SIZE)
        inp = torch.cat([obs, goal, self.coordconv.expand(batch_size, 2, CANVAS_SIZE, CANVAS_SIZE)], dim=1)

        if target:
            value = self.critic_target_net(inp)
        else:
            value = self.critic_net(inp)

        return value

    def optimize(self):
        '''Updates policy and value network using a batch from the replay buffer'''

        # Sample a batch
        obs, action, reward, next_obs, done, goal = self.replay_buffer.sample_batch(self.batch_size)
        obs = obs.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_obs = next_obs.to(self.device)
        done = done.to(self.device)
        goal = goal.to(self.device)

        # Update critic
        critic_estimation = self.evaluate_state(obs, goal, target=False)
        with torch.no_grad():
            # This update is different from the LearningToPaint paper (which does one extra step ahead for some reason).
            # Need to check if this is actually correct.
            improved_estimation = reward + self.discount * self.evaluate_state(next_obs, goal, target=True) * (1 - done.float())
        critic_loss = (improved_estimation - critic_estimation).pow(2).mean()
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # Update actor
        actor_action = self.select_action(obs, goal, target=False)
        resulting_obs = []
        for i in range(self.batch_size):
            resulting_obs.append(self.env.get_next_observation(obs[i], actor_action[i]))
        resulting_obs = torch.stack(resulting_obs, dim=0)
        actor_loss = -self.evaluate_state(resulting_obs, goal, target=True).mean() # This also seems different (they don't use target network to evaluate the state?)
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # Update target networks
        soft_update(self.actor_target_net, self.actor_net, self.tau)
        soft_update(self.critic_target_net, self.critic_net, self.tau)

    def set_device(self, device):
        self.device = device
        self.actor_net.to(device)
        self.actor_target_net.to(device)
        self.critic_net.to(device)
        self.critic_target_net.to(device)
        self.coordconv = self.coordconv.to(device)
