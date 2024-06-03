from renderers.diff_path_renderer import DiffPathRenderer
from envs.clip_loss import clip_conv_loss
from constants import CANVAS_SIZE
import torch
import numpy as np
import os
import random
import cv2

class DiffBezierSharpieEnv:
    def __init__(self, device, batch_size, stroke_bundle_size):
        self.renderer = DiffPathRenderer()
        self.MAX_STEPS = 10
        self.device = device
        self.thickness = 20
        self.batch_size = batch_size
        self.stroke_param_dim = 6
        self.stroke_bundle_size = stroke_bundle_size
        self.action_dim = self.stroke_param_dim * stroke_bundle_size
        self.reset()

    def reset(self):
        self.canvas = torch.zeros((self.batch_size, CANVAS_SIZE, CANVAS_SIZE))
        self.canvas = self.canvas.to(self.device)
        self.cur_steps = 0

    def step(self, action):
        ''' action: batch_size x action_dim '''

        if self.cur_steps == self.MAX_STEPS:
            raise Exception("Tried to step terminated episode")

        action = action.view(self.batch_size*self.stroke_bundle_size, self.stroke_param_dim) # B*bundle x stroke_param_dim
        traj = self.action2traj(action) # B*bundle x POINTS_PER_TRAJECTORY x 2
        stroke = self.renderer(traj, thickness=self.thickness) # B*bundle x CANVAS_SIZE x CANVAS_SIZE
        stroke = stroke.view(self.batch_size, self.stroke_bundle_size, CANVAS_SIZE, CANVAS_SIZE) # B x bundle x CANVAS_SIZE x CANVAS_SIZE

        for i in range(self.stroke_bundle_size):
            # Apply ith stroke in bundle
            cur_stroke = stroke[:, i, :, :]

            # Alpha blending according to the over operator (https://en.wikipedia.org/wiki/Alpha_compositing)
            alpha_a = cur_stroke
            self.canvas = cur_stroke + self.canvas*(1-alpha_a)

        self.cur_steps += 1

        return self.get_observation(), self.cur_steps == self.MAX_STEPS

    def get_observation(self):
        canvas = self.get_canvas()
        steps_left = torch.ones((self.batch_size, CANVAS_SIZE, CANVAS_SIZE, 1)) * (1 - self.cur_steps/self.MAX_STEPS)
        steps_left = steps_left.to(self.device)
        observation = torch.concat([canvas, steps_left], dim=3)
        return observation
    
    def get_canvas(self):
        canvas = 1.0 - self.canvas
        canvas = torch.stack([canvas, canvas, canvas], dim=3)
        return canvas
    
    def get_next_observation(self, observation, action):
        '''
        Gets the next observation if you were to take the action.
        This function is differentiable wrt action.
        '''
        canvas = 1.0 - observation[:,:,:,0]      # B x CANVAS_SIZE x CANVAS_SIZE
        steps_left = observation[:,:,:,3]        # B x CANVAS_SIZE x CANVAS_SIZE

        action = action.view(self.batch_size*self.stroke_bundle_size, self.stroke_param_dim) # B*bundle x stroke_param_dim
        traj = self.action2traj(action) # B*bundle x POINTS_PER_TRAJECTORY x 2
        stroke = self.renderer(traj, thickness=self.thickness) # B*bundle x CANVAS_SIZE x CANVAS_SIZE
        stroke = stroke.view(self.batch_size, self.stroke_bundle_size, CANVAS_SIZE, CANVAS_SIZE) # B x bundle x CANVAS_SIZE x CANVAS_SIZE

        # Alpha blending
        for i in range(self.stroke_bundle_size):
            cur_stroke = stroke[:, i, :, :]
            alpha_a = cur_stroke
            canvas = cur_stroke + canvas*(1-alpha_a)

        steps_left = steps_left - 1/self.MAX_STEPS

        return torch.stack([1.0-canvas, 1.0-canvas, 1.0-canvas, steps_left], dim=3)
    
    def calc_reward(self, prev_obs, cur_obs, goal_canvas):
        '''
        prev_obs: B x CANVAS_SIZE x CANVAS_SIZE x 4
        cur_obs: B x CANVAS_SIZE x CANVAS_SIZE x 4
        goal_canvas: B x CANVAS_SIZE x CANVAS_SIZE x 3
        '''
        def calc_score(canvas, goal_canvas):
            # canvas = torch.mean(canvas, dim=3) # B x CANVAS_SIZE x CANVAS_SIZE
            # goal_canvas = torch.mean(goal_canvas, dim=3) # B x CANVAS_SIZE x CANVAS_SIZE

            # black_mask = (goal_canvas <= 0.2).float() # B x CANVAS_SIZE x CANVAS_SIZE
            # white_mask = 1 - black_mask # B x CANVAS_SIZE x CANVAS_SIZE
            # l2 = ((canvas - goal_canvas) ** 2) # B x CANVAS_SIZE x CANVAS_SIZE
            # black_loss = (l2 * black_mask).mean(1).mean(1) # B
            # white_loss = (l2 * white_mask).mean(1).mean(1) # B
            # return 0.9 * black_loss + 0.1 * white_loss
            B = canvas.shape[0]
            res = []
            for i in range(B):
                this_canvas, this_goal = canvas[i:i+1], goal_canvas[i:i+1] # B x CANVAS_SIZE x CANVAS_SIZE x 3
                this_canvas = torch.permute(this_canvas, (0, 3, 1, 2)) # B x 3 x CANVAS_SIZE x CANVAS_SIZE
                this_goal = torch.permute(this_goal, (0, 3, 1, 2)) # B x 3 x CANVAS_SIZE x CANVAS_SIZE
                loss = clip_conv_loss(this_canvas, this_goal)
                res.append(loss)
            res = torch.stack(res)
            return res

        prev_canvas = prev_obs[:,:,:,:3]
        cur_canvas = cur_obs[:,:,:,:3]

        prev_score = calc_score(prev_canvas, goal_canvas)
        cur_score = calc_score(cur_canvas, goal_canvas)
        
        return prev_score - cur_score
    
    def action2traj(self, action):
        # action: n x 6
        n = len(action)
        length, bend1, bend2, start_x, start_y, theta = action[:,0], action[:,1], action[:,2], action[:,3], action[:,4], action[:,5]
        bend1 = bend1*0.2 - 0.1
        bend2 = bend2*0.2 - 0.1
        length = length * 0.5 + 0.05
        start_x = start_x * 0.8 + 0.1
        start_y = start_y * 0.8 + 0.1
        theta = theta * 2*np.pi

        zero = torch.zeros(n).to(action.device)
        p0 = torch.stack([zero, zero], dim=1)
        p1 = torch.stack([length/3, bend1], dim=1)
        p2 = torch.stack([length/3*2, bend2], dim=1)
        p3 = torch.stack([length, zero], dim=1)

        points = torch.stack([p0, p1, p2, p3], dim=1) # n x 4 x 2

        # Rotate and translate
        rot = torch.stack([
            torch.stack([torch.cos(theta), -torch.sin(theta)], dim=1),
            torch.stack([torch.sin(theta), torch.cos(theta)], dim=1)
        ], dim=1) # n x 2 x 2
        rot = torch.transpose(rot, 1, 2)
        points = points @ rot # n x 4 x 2
        
        trans = torch.stack([start_x, start_y], dim=1).unsqueeze(1) # n x 1 x 2
        points = points + trans # n x 4 x 2

        traj = []
        POINTS_PER_TRAJECTORY = 8
        for i in range(POINTS_PER_TRAJECTORY):
            t = i / (POINTS_PER_TRAJECTORY - 1)
            pos = (1-t)**3 * points[:,0] + 3*(1-t)**2 * t * points[:,1] + 3*(1-t) * t**2 * points[:,2] + t**3 * points[:,3] # n x 2
            traj.append(pos)
        traj = torch.stack(traj, dim=1) # n x POINTS_PER_TRAJECTORY x 2

        return traj
