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

def sample_image(data_dir):
    files = os.listdir(data_dir)
    file = random.choice(files)
    path = os.path.join(data_dir, file)

    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (CANVAS_SIZE, CANVAS_SIZE))
    img = torch.from_numpy(img) / 255

    return img

def main(args):
    wandb.init(project="RLFrida", config=args, mode=args.wandb_mode)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = DiffBezierSharpieEnv(device)
    agent = DDPG(env)
    agent.set_device(device)

    wandb.watch(agent.actor_net, log_freq=10, log='all')

    for episode_idx in range(args.num_episodes):
        # goal = sample_image(args.data_dir).to(device)
        goal = torch.zeros((CANVAS_SIZE, CANVAS_SIZE, 3))
        # goal[:CANVAS_SIZE//2, :, :] = 0
        goal = goal.to(device)
        env.reset()

        tot_reward = 0
        traj = []
        with torch.no_grad():
            while True:
                obs = env.get_observation()
                action = agent.select_action(obs, goal)
                next_obs, done = env.step(action)
                reward = env.calc_reward(obs, next_obs, goal)
                tot_reward += reward

                traj.append((obs, action, reward, next_obs, done, goal))

                if done:
                    break
        
        hindsight_goal = env.get_canvas()
        wandb.log({
            "episode": episode_idx,
            "total reward": tot_reward,
            "goal": wandb.Image(goal.cpu().numpy()),
            "hindsight goal": wandb.Image(hindsight_goal.cpu().numpy()),
        })

        for obs, action, reward, next_obs, done, goal in traj:
            agent.add_to_replay_buffer(
                obs,
                action,
                reward,
                next_obs,
                done,
                goal
            )
            # agent.add_to_replay_buffer(
            #     obs,
            #     action,
            #     env.calc_reward(obs, next_obs, hindsight_goal),
            #     next_obs,
            #     done,
            #     hindsight_goal
            # )
        
        if episode_idx >= args.num_warmup_episodes:
            agent.optimize()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_episodes', type=int, default=5000)
    parser.add_argument('--num_warmup_episodes', type=int, default=10)
    parser.add_argument('--data_dir', type=str, default=r"C:\Users\Ldori\OneDrive\Desktop\RLFrida\data\ContourDrawingDataset")
    parser.add_argument('--wandb_mode', type=str, default="disabled")

    args = parser.parse_args()
    main(args)
