import argparse
from ddpg.ddpg import DDPG
from envs.diff_bezier_sharpie_env import DiffBezierSharpieEnv
from constants import CANVAS_SIZE
import os
import random
import torch
import cv2

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = DDPG(device)
    env = DiffBezierSharpieEnv()

    for episode_idx in range(args.num_warmup_episodes):
        goal = sample_image(args.data_dir).to(device)
        env.reset()

        traj = []
        while True:
            canvas = env.observe()
            action = agent.select_action(canvas)
            next_canvas, done = env.step(action)
            reward = env.calc_reward(canvas, next_canvas, goal)

            traj.append((canvas, action, reward, next_canvas, done))

            if done:
                break
        
        hindsight_goal = env.observe()

        for canvas, action, reward, next_canvas, done in traj:
            agent.add_to_replay_buffer(
                torch.cat([canvas, goal], dim=2),
                action,
                reward,
                torch.cat([next_canvas, goal], dim=2),
                done
            )
            agent.add_to_replay_buffer(
                torch.cat([canvas, hindsight_goal], dim=2),
                action,
                env.calc_reward(canvas, next_canvas, hindsight_goal),
                torch.cat([next_canvas, hindsight_goal], dim=2),
                done
            )
        
        agent.optimize()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_episodes', type=int, default=1000)
    parser.add_argument('--num_warmup_episodes', type=int, default=10)
    parser.add_argument('--data_dir', type=str, default=r"C:\Users\Ldori\OneDrive\Desktop\RLFrida\data\ContourDrawingDataset")

    args = parser.parse_args()
    main(args)
