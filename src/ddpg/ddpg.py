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

        self.replay_buffer = ReplayBuffer(1000)

        self.env = env

        self.device = 'cpu'

        self.coordconv = torch.zeros((1, 2, CANVAS_SIZE, CANVAS_SIZE))
        for i in range(CANVAS_SIZE):
            for j in range(CANVAS_SIZE):
                self.coordconv[0, 0, i, j] = i / CANVAS_SIZE
                self.coordconv[0, 1, i, j] = j / CANVAS_SIZE

        self.actor_net = ResNet_wobn(9, 18, self.env.action_dim) # target, canvas, stepnum, coordconv 3 + 3 + 1 + 2
        self.actor_target_net = ResNet_wobn(9, 18, self.env.action_dim)
        self.critic_net = ResNet_wobn(9, 18, 1)
        self.critic_target_net = ResNet_wobn(9, 18, 1)

        self.actor_optim = optim.Adam(self.actor_net.parameters(), lr=1e-5)
        self.critic_optim = optim.Adam(self.critic_net.parameters(), lr=1e-5)

        hard_update(self.actor_target_net, self.actor_net)
        hard_update(self.critic_target_net, self.critic_net)

    def add_to_replay_buffer(self, obs, action, reward, next_obs, done, goal):
        '''Store the transition into the replay buffer'''
        obs = obs.cpu().detach()
        action = action.cpu().detach()
        reward = reward.cpu().detach()
        next_obs = next_obs.cpu().detach()
        done = torch.tensor(done, dtype=torch.float)
        goal = goal.cpu().detach()

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
    
    def l2p_play(self, obs, goal, target=False):
        action = self.select_action(obs, goal, target)
        return action

    def l2p_evaluate(self, obs, goal, action, target=False):
        next_obs = []
        rew = []
        for i in range(obs.shape[0]):
            next_obs_ = self.env.get_next_observation(obs[i], action[i])
            rew_ = self.env.calc_reward(obs[i], next_obs_, goal[i])
            next_obs.append(next_obs_)
            rew.append(rew_)
        next_obs = torch.stack(next_obs)
        rew = torch.stack(rew)

        Q = self.evaluate_state(next_obs, goal, target)

        return (Q + rew), rew
    
    def optimize(self):
        self.train()
        '''Updates policy and value network using a batch from the replay buffer'''

        # Sample a batch
        obs, action, reward, next_obs, done, goal = self.replay_buffer.sample_batch(self.batch_size)
        obs = obs.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_obs = next_obs.to(self.device)
        done = done.to(self.device)
        goal = goal.to(self.device)

        # import matplotlib.pyplot as plt
        # print("optimize called")
        # for i in range(5):
        #     _, axs = plt.subplots(1, 3)
        #     axs[0].imshow(obs[i,:,:,:3].cpu().numpy())
        #     axs[1].imshow(next_obs[i,:,:,:3].cpu().numpy())
        #     axs[2].imshow(goal[i].cpu().numpy())
        #     print(action[i], reward[i], done[i])
        #     plt.show()

        # Update critic
        # critic_estimation = self.evaluate_state(obs, goal, target=False)
        # with torch.no_grad():
        #     # This update is different from the LearningToPaint paper (which does one extra step ahead for some reason).
        #     # Need to check if this is actually correct.
        #     improved_estimation = reward + self.discount * self.evaluate_state(next_obs, goal, target=True) * (1 - done.float())
        # critic_loss = (improved_estimation - critic_estimation).pow(2).mean()
        # self.critic_optim.zero_grad()
        # critic_loss.backward()
        # self.critic_optim.step()

        # Update actor
        # actor_action = self.select_action(obs, goal, target=False)
        # resulting_obss = []
        # resulting_rewards = []
        # for i in range(self.batch_size):
        #     resulting_obs = self.env.get_next_observation(obs[i], actor_action[i])
        #     resulting_reward = self.env.calc_reward(obs[i], resulting_obs, goal[i])

        #     resulting_obss.append(resulting_obs)
        #     resulting_rewards.append(resulting_reward)
        # resulting_obss = torch.stack(resulting_obss, dim=0)
        # resulting_rewards = torch.stack(resulting_rewards, dim=0)
        # estimated_values = resulting_rewards + self.discount * self.evaluate_state(resulting_obss, goal, target=True) # This also seems different (they don't use target network to evaluate the state?)
        # actor_loss = -estimated_values.mean()
        # self.actor_optim.zero_grad()
        # actor_loss.backward()
        # self.actor_optim.step()

        # L2P critic update
        # with torch.no_grad():
        #     next_action = self.l2p_play(next_obs, goal, target=True)
        #     target_q, _ = self.l2p_evaluate(next_obs, goal, next_action, target=True)
        #     target_q = self.discount * (1-done.float()) * target_q
        # cur_q, step_reward = self.l2p_evaluate(obs, goal, action)
        # target_q += step_reward.detach()

        # value_loss = ((target_q - cur_q)**2).mean()
        # self.critic_net.zero_grad()
        # value_loss.backward(retain_graph=True)
        # self.critic_optim.step()
        
        # L2P actor update
        # action = self.l2p_play(obs, goal)
        # pre_q, _ = self.l2p_evaluate(obs.detach(), goal.detach(), action)
        # policy_loss = -pre_q.mean()
        action = self.select_action(obs, goal, target=False)
        next_obs = []
        rew = []
        for i in range(obs.shape[0]):
            next_obs_ = self.env.get_next_observation(obs[i], action[i])
            next_obs.append(next_obs_)
            rew_ = self.env.calc_reward(obs[i], next_obs_, goal[i])
            rew.append(rew_)
        next_obs = torch.stack(next_obs)
        rew = torch.stack(rew)
        
        # rew = -torch.mean((action[:, 3] - obs[:, 0, 0, 3])**2)
        # rew = -torch.mean(next_obs[:,:,:,:3])
        rew = torch.mean(rew)
        print(rew)

        policy_loss = -rew
        self.actor_net.zero_grad()
        policy_loss.backward(retain_graph=True)
        self.actor_optim.step()

        # Update target networks
        soft_update(self.actor_target_net, self.actor_net, self.tau)
        soft_update(self.critic_target_net, self.critic_net, self.tau)
        self.eval()

    def set_device(self, device):
        self.device = device
        self.actor_net.to(device)
        self.actor_target_net.to(device)
        self.critic_net.to(device)
        self.critic_target_net.to(device)
        self.coordconv = self.coordconv.to(device)
    
    def train(self):
        self.actor_net.train()
        self.actor_target_net.train()
        self.critic_net.train()
        self.critic_target_net.train()
    
    def eval(self):
        self.actor_net.eval()
        self.actor_target_net.eval()
        self.critic_net.eval()
        self.critic_target_net.eval()

