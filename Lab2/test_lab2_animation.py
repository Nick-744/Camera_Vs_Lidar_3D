from vvrpywork.constants import Color
from vvrpywork.scene import Scene2D
from vvrpywork.shapes import (
    Point2D, Line2D, Rectangle2D,
    PointSet2D, LineSet2D, Polygon2D
)

from time import sleep
import pyglet

WIDTH = 800
HEIGHT = 800

class Lab2(Scene2D):
    def __init__(self):
        super().__init__(WIDTH, HEIGHT, "Lab2")
        self.reset()

        return;

    def reset(self):
        bound = Rectangle2D((-0.5, -0.5), (0.5, 0.5))
        self.pcd = PointSet2D(color = Color.RED)
        self.pcd.createRandom(bound, 8, color = Color.BLUE)
        self.addShape(self.pcd, "pcd")

        self.tempPoints2D = [Point2D(pt) for pt in self.pcd.points]
        self.point_pairs = [
            (p1, p2)
            for p1 in self.tempPoints2D
            for p2 in self.tempPoints2D
            if self.checkSamePoint(p1, p2) == False
        ]
        self.lineset = LineSet2D()

        self.current_index = 0
        self.helper_index = 0
        self.current_line = None
        self.line_valid = True

        pyglet.clock.schedule_interval(self.animate, 1/3)

        return;

    def animate(self, dt):
        try: # Για να αποφύγει το IndexError: list index out of range στην τελευταία επανάληψη!
            (p1, p2) = self.point_pairs[self.current_index]
        except:
            pass

        if self.current_line is None:
            if self.current_index >= len(self.point_pairs): # Τέλος η ζωγραφική
                pyglet.clock.unschedule(self.animate)

                self.poly = Polygon2D.create_from_lineset(self.lineset, color=Color.CYAN)
                self.addShape(self.poly, "ch")

                return;

            self.current_line = Line2D(p1, p2, color = Color.GREEN)
            self.addShape(self.current_line, f"line_{self.current_index}")

            (p1.size, p1.color) = (1.5, Color.BLACK)
            (p2.size, p2.color) = (1, Color.BLACK)
            self.addShape(p1, f"p1_{self.current_index}")
            self.addShape(p2, f"p2_{self.current_index}")

            self.helper_index = 0
            self.line_valid = True

        if self.helper_index >= len(self.tempPoints2D):
            if self.line_valid: # Η γραμμή ανήκει στο κυρτό περίβλημα!
                self.lineset.add(self.current_line)
                self.current_line.color = Color.BLACK
                self.updateShape(f"line_{self.current_index}")
            else:
                self.removeShape(f"line_{self.current_index}")
                
            for i in range(self.helper_index):
                self.removeShape(f"p_{i}")

            self.removeShape(f"p1_{self.current_index}")
            self.removeShape(f"p2_{self.current_index}")

            self.current_line = None
            self.current_index += 1

            return;

        p3 = self.tempPoints2D[self.helper_index]
        if not self.checkSamePoint(p3, p1) and not self.checkSamePoint(p3, p2):
            (p3.size, p3.color) = (1.5, Color.GREEN)
            self.addShape(p3, f"p_{self.helper_index}")

            if self.current_line.isOnRight(p3):
                self.current_line.color = Color.RED
                self.updateShape(f"line_{self.current_index}")

                p3.color = Color.RED
                self.updateShape(f"p_{self.helper_index}")

                self.line_valid = False
                self.helper_index = len(self.tempPoints2D) # Cheat ώστε να ελέγξει νέα γραμμή!

        self.helper_index += 1

        return;

    def checkSamePoint(self, p1, p2):
        return p1.x == p2.x and p1.y == p2.y;

if __name__ == "__main__":
    app = Lab2()
    sleep(10)
    app.mainLoop()
