import torch
from torch import nn
from constants import CANVAS_SIZE

class Traj2Stroke(nn.Module):
    def __init__(self, n_pts):
        super(Traj2Stroke, self).__init__()
        
        self.n_pts = n_pts

        self.thickness = nn.Parameter(torch.zeros(1))
        self.darkness_dropoff = nn.Parameter(torch.zeros(1))

        idxs_x = torch.arange(CANVAS_SIZE)
        idxs_y = torch.arange(CANVAS_SIZE)
        x_coords, y_coords = torch.meshgrid(idxs_y, idxs_x, indexing='ij') # CANVAS_SIZE x CANVAS_SIZE
        self.grid_coords = torch.stack((y_coords, x_coords), dim=2).reshape(1,CANVAS_SIZE,CANVAS_SIZE,2) # 1 x CANVAS_SIZE x CANVAS_SIZE x 2

    def forward(self, traj):
        # traj: (self.n_pts, 2)
        traj = traj * CANVAS_SIZE
        n = len(traj)
        vs = traj[:-1].reshape((-1,1,1,2)) # (P-1, 1, 1, 2)
        vs = torch.tile(vs, (1, CANVAS_SIZE, CANVAS_SIZE, 1)) # (P-1, CANVAS_SIZE, CANVAS_SIZE, 2)

        ws = traj[1:].reshape((-1,1,1,2)) # (P-1, 1, 1, 2)
        ws = torch.tile(ws, (1, CANVAS_SIZE, CANVAS_SIZE, 1)) # (P-1, CANVAS_SIZE, CANVAS_SIZE, 2)

        coords = torch.tile(self.grid_coords, (n-1,1,1,1)).to(ws.device) # (P-1, CANVAS_SIZE, CANVAS_SIZE, 2)

        # For each of the P segments, compute distance from every point to the line
        def dist_line_segment(p, v, w):
            d = torch.linalg.norm(v-w, dim=3) # (n-1) x CANVAS_SIZE x CANVAS_SIZE
            dot = (p-v) * (w-v)
            dot_sum = torch.sum(dot, dim=3) / (d**2 + 1e-5)
            t = dot_sum.unsqueeze(3) # (n-1) x CANVAS_SIZE x CANVAS_SIZE x 1
            t = torch.clamp(t, min=0, max=1) # (n-1) x CANVAS_SIZE x CANVAS_SIZE x 1
            proj = v + t * (w-v) # (n-1) x CANVAS_SIZE x CANVAS_SIZE x 2
            return torch.linalg.norm(p-proj, dim=3)
        distances = dist_line_segment(coords, vs, ws) # (P-1, CANVAS_SIZE, CANVAS_SIZE)
        distances = torch.min(distances, dim=0).values

        thickness = 10 * torch.sigmoid(self.thickness)
        darkness = torch.clamp((thickness - distances) / thickness, min=0.0, max=1.0)
        dark_exp = 10 * torch.sigmoid(self.darkness_dropoff)
        darkness = darkness ** dark_exp

        return 1.0-darkness
