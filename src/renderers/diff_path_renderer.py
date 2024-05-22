import torch
from torch import nn
from constants import CANVAS_SIZE

class DiffPathRenderer(nn.Module):
    def __init__(self):
        super(DiffPathRenderer, self).__init__()

        idxs_x = torch.arange(CANVAS_SIZE)
        idxs_y = torch.arange(CANVAS_SIZE)
        x_coords, y_coords = torch.meshgrid(idxs_y, idxs_x, indexing='ij') # CANVAS_SIZE x CANVAS_SIZE
        self.grid_coords = torch.stack((y_coords, x_coords), dim=2).reshape(1,CANVAS_SIZE,CANVAS_SIZE,2) # 1 x CANVAS_SIZE x CANVAS_SIZE x 2

    def forward(self, traj, thickness):
        traj = traj * CANVAS_SIZE
        n = len(traj)

        vs = traj[:-1].reshape((-1,1,1,2)) # (n-1, 1, 1, 2)
        vs = torch.tile(vs, (1, CANVAS_SIZE, CANVAS_SIZE, 1)) # (n-1, CANVAS_SIZE, CANVAS_SIZE, 2)

        ws = traj[1:].reshape((-1,1,1,2)) # (n-1, 1, 1, 2)
        ws = torch.tile(ws, (1, CANVAS_SIZE, CANVAS_SIZE, 1)) # (n-1, CANVAS_SIZE, CANVAS_SIZE, 2)

        coords = torch.tile(self.grid_coords, (n-1,1,1,1)).to(ws.device) # (n-1, CANVAS_SIZE, CANVAS_SIZE, 2)

        # For each of the n-1 segments, compute distance from every point to the line
        def dist_line_segment(p, v, w):
            d = torch.linalg.norm(v-w, dim=3) # (n-1) x CANVAS_SIZE x CANVAS_SIZE
            dot = (p-v) * (w-v)
            dot_sum = torch.sum(dot, dim=3) / (d**2 + 1e-5)
            t = dot_sum.unsqueeze(3) # (n-1) x CANVAS_SIZE x CANVAS_SIZE x 1
            t = torch.clamp(t, min=0, max=1) # (n-1) x CANVAS_SIZE x CANVAS_SIZE x 1
            proj = v + t * (w-v) # (n-1) x CANVAS_SIZE x CANVAS_SIZE x 2
            return torch.linalg.norm(p-proj, dim=3)
        distances = dist_line_segment(coords, vs, ws) # (n-1, CANVAS_SIZE, CANVAS_SIZE)
        distances = torch.min(distances, dim=0).values

        radius = thickness/2
        darkness = torch.clamp((radius - distances) / radius, min=0.0, max=1.0)
        # darkness = darkness ** 0.2

        return darkness
