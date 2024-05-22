import argparse
from ddpg.ddpg import DDPG
from envs.diff_bezier_sharpie_env import DiffBezierSharpieEnv
from constants import CANVAS_SIZE
import os
import random
import torch
import cv2
import matplotlib.pyplot as plt
import wandb
import numpy as np

def preload_images(data_dir):
    files = os.listdir(data_dir)
    imgs = []
    for file in files:
        path = os.path.join(data_dir, file)

        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (CANVAS_SIZE, CANVAS_SIZE))
        img = torch.from_numpy(img) / 255
        imgs.append(img)
    return torch.stack(imgs, dim=0)

def sample_images(imgs, batch_size):
    idxs = np.random.choice(len(imgs), batch_size)
    return imgs[idxs]

def main(args):
    wandb.init(project="RLFrida", config=args, mode=args.wandb_mode)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = DiffBezierSharpieEnv(device, args.batch_size, args.stroke_bundle_size)
    agent = DDPG(env)
    agent.set_device(device)

    imgs = preload_images(args.data_dir)

    for episode_idx in range(args.num_episodes):
        goal = sample_images(imgs, args.batch_size).to(device)
        # goal = torch.ones((args.batch_size, CANVAS_SIZE, CANVAS_SIZE, 3))
        # goal[:CANVAS_SIZE//2, :, :] = 0
        goal = goal.to(device)
        env.reset()

        tot_reward = torch.zeros(args.batch_size).to(device)
        traj = []
        with torch.no_grad():
            while True:
                obs = env.get_observation()                           # B x CANVAS_SIZE x CANVAS_SIZE x 4
                action = agent.select_action(obs, goal)               # B x action_dim
                next_obs, done = env.step(action)                     # B x CANVAS_SIZE x CANVAS_SIZE x 4; 1
                done_tensor = torch.ones(args.batch_size)*float(done) # B
                reward = env.calc_reward(obs, next_obs, goal)         # B
                tot_reward += reward

                traj.append((obs, action, reward, next_obs, done_tensor, goal))

                if done:
                    break
        
        hindsight_goal = env.get_canvas() # B x CANVAS_SIZE x CANVAS_SIZE x 3

        wandb_dict = {
            "episode": episode_idx,
            "return": tot_reward.mean()
        }
        if episode_idx % 10 == 0:
            for i in range(min(10, args.batch_size)):
                wandb_dict[f"{i}/goal"] = wandb.Image(goal[i].cpu().numpy())
                wandb_dict[f"{i}/hindsight goal"] = wandb.Image(hindsight_goal[i].cpu().numpy())
        wandb.log(wandb_dict)

        for obs, action, reward, next_obs, done, goal in traj:
            agent.add_to_replay_buffer(
                obs,
                action,
                reward,
                next_obs,
                done,
                goal
            )
            agent.add_to_replay_buffer(
                obs,
                action,
                env.calc_reward(obs, next_obs, hindsight_goal),
                next_obs,
                done,
                hindsight_goal
            )
        
        if episode_idx >= args.num_warmup_episodes:
            for i in range(args.train_iters_per_episode):
                agent.optimize()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_episodes', type=int, default=5000)
    parser.add_argument('--num_warmup_episodes', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=96)
    parser.add_argument('--stroke_bundle_size', type=int, default=5)
    parser.add_argument('--data_dir', type=str, default=r"/home/frida/lawrence/RLFrida/data/ContourDrawingDataset")
    parser.add_argument('--wandb_mode', type=str, default="disabled")
    parser.add_argument('--train_iters_per_episode', type=int, default=10)

    args = parser.parse_args()
    main(args)
