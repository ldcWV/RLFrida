from constants import CANVAS_SIZE
import numpy as np
import cairo
import math

class SharpieRenderer():
    def __init__(self):
        self.canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE, 4), dtype=np.uint8)
        surface = cairo.ImageSurface.create_for_data(self.canvas, cairo.FORMAT_ARGB32, CANVAS_SIZE, CANVAS_SIZE)
        self.cr = cairo.Context(surface)
        self.clear()

    def draw(self, trajectory):
        path = []
        for i in range(len(trajectory)):
            path.append((
                CANVAS_SIZE * trajectory[i][0],
                CANVAS_SIZE * trajectory[i][1]
            ))

        radius = 2
        # draw trajectory path
        for i in range(len(path)-1):
            x1, y1 = path[i][0], path[i][1]
            x2, y2 = path[i+1][0], path[i+1][1]
            self.cr.set_line_width(2*radius)
            self.cr.move_to(x1, y1)
            self.cr.line_to(x2, y2)
            self.cr.set_source_rgb(0, 0, 0)
            self.cr.stroke()
        
        # draw circles along trajectory
        for i in range(len(path)):
            x, y = path[i][0], path[i][1]
            self.cr.arc(x, y, radius, 0, 2*math.pi)
            self.cr.set_source_rgb(0, 0, 0)
            self.cr.fill()
        
        return self.canvas[:,:,:3] / 255

    def clear(self):
        self.cr.set_source_rgb(255, 255, 255)
        self.cr.paint()

        return self.canvas[:,:,:3] / 255
