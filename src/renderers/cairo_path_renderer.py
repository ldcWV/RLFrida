from constants import CANVAS_SIZE
import numpy as np
import cairo
import math

class CairoPathRenderer():
    def __init__(self):
        self.canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8)
        surface = cairo.ImageSurface.create_for_data(self.canvas, cairo.FORMAT_A8, CANVAS_SIZE, CANVAS_SIZE)
        self.cr = cairo.Context(surface)

    def draw(self, trajectory, thickness):
        self.cr.set_source_rgba(0.0, 0.0, 0.0, 0.0)
        self.cr.paint()

        path = []
        for i in range(len(trajectory)):
            path.append((
                CANVAS_SIZE * trajectory[i][0],
                CANVAS_SIZE * trajectory[i][1]
            ))

        radius = thickness/2
        # draw trajectory path
        for i in range(len(path)-1):
            x1, y1 = path[i][0], path[i][1]
            x2, y2 = path[i+1][0], path[i+1][1]
            self.cr.set_line_width(2*radius)
            self.cr.move_to(x1, y1)
            self.cr.line_to(x2, y2)
            self.cr.set_source_rgba(0.0, 0.0, 0.0, 1.0)
            self.cr.stroke()
        
        # draw circles along trajectory
        for i in range(len(path)):
            x, y = path[i][0], path[i][1]
            self.cr.arc(x, y, radius, 0, 2*math.pi)
            self.cr.set_source_rgba(0.0, 0.0, 0.0, 1.0)
            self.cr.fill()
        
        return self.canvas / 255
