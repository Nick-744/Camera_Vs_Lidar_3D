from vvrpywork.constants import Key, Mouse, Color
from vvrpywork.scene import Scene2D
from vvrpywork.shapes import (
    Point2D, Line2D, Triangle2D, Circle2D, Rectangle2D,
    PointSet2D, LineSet2D, Polygon2D, Label2D
)

import numpy as np
import pyglet

WIDTH = 800
HEIGHT = 800

class Lab2(Scene2D):
    def __init__(self):
        super().__init__(WIDTH, HEIGHT, "Lab2")

        self.point = Point2D((0, 0), color = Color.RED, size = 3)
        self.addShape(self.point, "point")

        self.direction = True
        self.border = 1

        pyglet.clock.schedule_interval(self.update, 1/60)

        return;
        
    def update(self, dt):
        if self.direction:
            self.point.x += 0.01
            self.point.y += 0.01
        else:
            self.point.x -= 0.01
            self.point.y -= 0.01
        
        if self.point.x > self.border:
            self.direction = False
        elif self.point.x < -self.border:
            self.direction = True

        self.updateShape("point")

        return;

    def on_key_press(self, symbol, modifiers):
        if symbol == Key.SPACE:
            self.direction = not self.direction
        
        return;

if __name__ == "__main__":
    lab2 = Lab2()
    lab2.mainLoop()