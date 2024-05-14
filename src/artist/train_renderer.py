import sys
import os
sys.path.append(os.path.abspath('..'))

import torch
from gt_renderers.sharpie_renderer import SharpieRenderer
from artist.traj2stroke import Traj2Stroke
from artist.config import POINTS_PER_TRAJECTORY, MIN_LENGTH, MAX_LENGTH, MAX_BEND
import math
import random
import numpy as np
import matplotlib.pyplot as plt

NUM_ITERS = 500
BATCH_SIZE = 8
PLOT_RESULTS = False

def sample_trajectory():
    length = random.random() * (MAX_LENGTH - MIN_LENGTH) + MIN_LENGTH
    bend1 = (random.random()*2-1) * MAX_BEND
    bend2 = (random.random()*2-1) * MAX_BEND
    start = np.random.rand(2)
    theta = random.random() * 2 * math.pi

    p0 = np.array([0.0, 0.0])
    p1 = np.array([length/3, bend1])
    p2 = np.array([length/3*2, bend2])
    p3 = np.array([length, 0.0])

    points = np.array([p0, p1, p2, p3])
    
    # Rotate and translate
    points = points @ np.array([
        [math.cos(theta), -math.sin(theta)],
        [math.sin(theta), math.cos(theta)]
    ]).T
    points = points + start

    traj = []
    for i in range(POINTS_PER_TRAJECTORY):
        t = i / (POINTS_PER_TRAJECTORY - 1)
        pos = (1-t)**3 * points[0] + 3*(1-t)**2 * t * points[1] + 3*(1-t) * t**2 * points[2] + t**3 * points[3]
        traj.append(pos)
    traj = np.array(traj)

    return traj

def main():
    gt_renderer = SharpieRenderer()
    model = Traj2Stroke(POINTS_PER_TRAJECTORY)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    optim = torch.optim.Adam(model.parameters(), lr=0.1)

    for i in range(NUM_ITERS):
        tot_loss = 0
        for j in range(BATCH_SIZE):
            traj = sample_trajectory()

            gt_renderer.clear()
            gt_stroke = gt_renderer.draw(traj)
            gt_stroke = torch.from_numpy(gt_stroke).to(device)

            traj = torch.from_numpy(traj).to(device)
            pred_stroke = model(traj)
            pred_stroke = torch.stack([pred_stroke, pred_stroke, pred_stroke], dim=2)

            if PLOT_RESULTS and i == NUM_ITERS-1 and j >= BATCH_SIZE-3:
                _, axs = plt.subplots(1,2)
                axs[0].imshow(gt_stroke.cpu().detach().numpy())
                axs[1].imshow(pred_stroke.cpu().detach().numpy())
                plt.show()

            loss = torch.mean((gt_stroke - pred_stroke)**2)
            tot_loss += loss
        avg_loss = tot_loss / BATCH_SIZE

        optim.zero_grad()
        avg_loss.backward()
        optim.step()

        if i % 20 == 0:
            print(f"Epoch {i} loss: {avg_loss.item()}")
            for name, param in model.named_parameters():
                if param.requires_grad:
                    print(f"{name}: {param.data.item()}")
    
    # Save model
    torch.save(model.state_dict(), 'renderer.pt')

if __name__ == '__main__':
    main()
