from renderers.diff_path_renderer import DiffPathRenderer
from constants import CANVAS_SIZE
import torch
import numpy as np
import os
import random
import cv2

class DiffBezierSharpieEnv:
    def __init__(self):
        self.action_space = (5)
        self.renderer = DiffPathRenderer()
        self.MAX_STEPS = 100
        self.reset()

    def reset(self):
        self.canvas = torch.zeros((CANVAS_SIZE, CANVAS_SIZE))
        self.cur_steps = 0

    def step(self, action):
        if self.cur_steps == self.MAX_STEPS:
            raise Exception("Tried to step terminated episode")

        traj = self.action2traj(action)
        stroke = self.renderer(traj, thickness=1.5)
        self.canvas = torch.max(self.canvas, stroke)
        self.cur_steps += 1

        return self.get_observation(), self.cur_steps == self.MAX_STEPS

    def get_observation(self):
        canvas = self.get_canvas()
        steps_left = torch.ones((CANVAS_SIZE, CANVAS_SIZE)) * (self.MAX_STEPS - self.cur_steps)
        observation = torch.stack([canvas, steps_left], dim=2)
        return observation
    
    def get_canvas(self):
        canvas = 1.0 - self.canvas
        canvas = torch.stack([canvas, canvas, canvas], dim=2)
        return canvas
    
    def get_next_observation(self, observation, action):
        '''
        Gets the next observation if you were to take the action.
        This function is differentiable wrt action.
        '''
        prev_canvas = observation[:,:,:2]
        prev_steps_left = observation[:,:,2:]

        traj = self.action2traj(action)
        stroke = self.renderer(traj, thickness=1.5)
        stroke = stroke.unsqueeze(2)

        new_canvas = torch.min(prev_canvas, 1.0-stroke)
        new_steps_left = prev_steps_left - 1
        return torch.stack([new_canvas, new_steps_left], dim=2)
    
    def calc_reward(self, prev_obs, cur_obs, goal_canvas):
        prev_canvas = prev_obs[:,:,:2]
        cur_canvas = cur_obs[:,:,:2]

        prev_l2 = torch.linalg.norm(prev_canvas - goal_canvas, dim=2)
        cur_l2 = torch.linalg.norm(cur_canvas - goal_canvas, dim=2)
        
        return prev_l2 - cur_l2
    
    def action2traj(self, action):
        length, bend1, bend2, start, theta = action

        zero = torch.tensor(0.0).to(action.device)
        p0 = torch.stack([zero, zero])
        p1 = torch.stack([length/3, bend1])
        p2 = torch.stack([length/3*2, bend2])
        p3 = torch.stack([length, zero])

        points = torch.stack([p0, p1, p2, p3])

        # Rotate and translate
        rot = torch.stack([
            torch.stack([torch.cos(theta), -torch.sin(theta)]),
            torch.stack([torch.sin(theta), torch.cos(theta)])
        ])
        points = points @ rot.T
        points = points + start

        traj = []
        POINTS_PER_TRAJECTORY = 16
        for i in range(POINTS_PER_TRAJECTORY):
            t = i / (POINTS_PER_TRAJECTORY - 1)
            pos = (1-t)**3 * points[0] + 3*(1-t)**2 * t * points[1] + 3*(1-t) * t**2 * points[2] + t**3 * points[3]
            traj.append(pos)
        traj = torch.stack(traj)

        return traj
