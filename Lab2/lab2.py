from vvrpywork.constants import Key, Mouse, Color
from vvrpywork.scene import Scene2D
from vvrpywork.shapes import (
    Point2D, Line2D, Triangle2D, Circle2D, Rectangle2D,
    PointSet2D, LineSet2D, Polygon2D, Label2D
)

from time import time
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

WIDTH = 800
HEIGHT = 800

class Lab2(Scene2D):
    def __init__(self):
        super().__init__(WIDTH, HEIGHT, "Lab2")
        self.reset()
        #self.question5()

        return;

    def reset(self):
        self.my_mouse_pos = Point2D((0, 0))
        self.addShape(self.my_mouse_pos, "mouse")

        bound = Rectangle2D((-0.5, -0.5), (0.5, 0.5))
        self.pcd = PointSet2D(color=Color.RED)
        self.pcd.createRandom(bound, 50, "Lab2", Color.RED)
        self.addShape(self.pcd)

        #self.CH_brute_force()
        #self.CH_graham_scan()
        self.CH_quickhull()
        #self.CH_jarvis_march()

        return;
    
# ----- Ερώτημα 5ο -----

    def question5(self):
        foos = [self.CH_brute_force, self.CH_graham_scan, self.CH_quickhull, self.CH_jarvis_march]
        labels = ["Brute Force", "Graham Scan", "QuickHull", "Jarvis March"]
        #foos = [self.CH_graham_scan, self.CH_quickhull]
        #labels = ["Graham Scan", "QuickHull"]
        num_points = [10, 100, 500, 1000, 2000]
        #num_points = [100, 500, 1000, 5000, 10000, 25000, 50000, 100000, 250000, 500000]
        
        for foo, label in zip(foos, labels):
            if label == "Brute Force" and num_points[-1] > 2000: # Πολύ χρονοβόρος αλγόριθμος
                continue;

            times = []
            for n in num_points:
                (_, _, exec_time) = self.testFunctionSpeed(foo, n)
                times.append(exec_time)
            print() # Καλύτερη αισθητική στο output
        
            # Plot σε νέο παράθυρο
            fig, ax = plt.subplots()
            ax.set_title(f"Execution time for {label}")
            ax.set_xlabel("Number of points")
            ax.set_ylabel("Execution time (s)")
            
            ax.plot(num_points, times, marker = 'o', linestyle = '-')
            ax.grid(True, which="both", linestyle = "--", linewidth = 0.5)
            
            plt.show()

        return;

    def testFunctionSpeed(self, func, num_points, printVariable = True):
        # Δημιουργία δεδομένων δοκιμής
        bound = Rectangle2D((-0.5, -0.5), (0.5, 0.5))
        self.pcd = PointSet2D(color = Color.RED)
        self.pcd.createRandom(bound, num_points)
        self.addShape(self.pcd)

        start = time()

        func()

        stop = time()
        executionTime = stop - start

        if printVariable == True:
            print(f"Function's name:  {func.__name__}")
            print(f"Number of points: {num_points}")
            print(f"Execution time:   {executionTime}")

        return func.__name__, num_points, executionTime;

# ----- Υλοποίηση αλγορίθμων εΰρεσης κυρτού περιβλήματος -----

    def CH_brute_force(self):
        if len(self.pcd) < 3:
            raise ValueError("No convex hull can be defined with less than 3 points!")
        
        lineset = LineSet2D()
        def checkSamePoint(p1, p2):
            statement = False
            if p1.x == p2.x and p1.y == p2.y:
                statement = True
            
            return statement;

        # add all line segments that make up the convex hull to self.lineset
        tempPoints2D = [Point2D(p) for p in self.pcd.points]
        for p1 in tempPoints2D:
            for p2 in tempPoints2D:
                if checkSamePoint(p2, p1):
                    continue;

                line = Line2D(p1, p2)
                for p3 in tempPoints2D:
                    if checkSamePoint(p3, p2) or checkSamePoint(p3, p1):
                        continue;
                    
                    if line.isOnRight(p3):
                        break;
                else:
                    lineset.add(line)
                
        # convert the lineset to a polygon
        self.poly = Polygon2D.create_from_lineset(lineset, color = Color.CYAN)
        self.addShape(self.poly, "ch")
        
        return;

    def CH_graham_scan(self):
        if len(self.pcd) < 3:
            raise ValueError("No convex hull can be defined with less than 3 points!")
        
        def orientation(p, q, r):
            return np.cross(q - p, r - q);

        #initial step
        # sorting according to angle
        p0_idx = np.argmin(self.pcd.points[:, 1])
        p0 = self.pcd.points[p0_idx]

        # Calculate the polar angle of each point with respect to p0
        angles = np.arctan2(self.pcd.points[:, 1] - p0[1], self.pcd.points[:, 0] - p0[0])

        # Sort the points by angle
        sorted_idx = np.argsort(angles)
        sorted_points = self.pcd.points[sorted_idx]

        stack = [0, 1] # Η λίστα πρχ δείκτες! Σίγουρα το 0 και 1, σκέψου το γιατί!
        ids = np.arange(2, sorted_points.shape[0])

        #iterating over the rest of the points
        for id in ids:
            #pop elements from the stack until graham condition is satisfied
            while len(stack) > 1 and not orientation(sorted_points[stack[-2]], sorted_points[stack[-1]], sorted_points[id]) > 0:
                stack.pop()

            #append current point
            stack.append(id)

        self.poly = Polygon2D(sorted_points[np.array(stack)], color = Color.GREEN)
        # [sorted_points[index] for index in stack] == sorted_points[np.array(stack)]
        self.addShape(self.poly, "ch")

        return;

    def CH_quickhull(self):
        if len(self.pcd) < 3:
            raise ValueError("No convex hull can be defined with less than 3 points!")

        def query_line(pointcloud, A, B):
            nonlocal chull # pass by ref

            #base case, empty pointcloud
            if pointcloud.shape[0] == 0:
                return;

            #--------------------------------------------------------------------------
            # finding the furthest point C from the line A B
            #--------------------------------------------------------------------------

            #projecting 
            AB = B - A
            AP = pointcloud - A # Βρες όλα τα ευθύγραμμα τμήματα που περιέχουνε ως αρχή το Α
            proj = AB * (AP @ AB.reshape(-1, 1)) / np.dot(AB, AB)

            #finding distances between points and their projection (which is the distance from the line)
            dist = np.linalg.norm(AP - proj, axis = 1)

            #the furthest point is the one with the maximum distance
            C = pointcloud[np.argmax(dist)]

            #adding C to the convex hull
            chull.append(C)

            #--------------------------------------------------------------------------
            # forming the lines CA, CB that constitute a triangle
            #--------------------------------------------------------------------------

            #separating the points on the right and on the left of AC
            ACleft, _ = separate_points_by_line(A, C, pointcloud)

            #separating the points on the right and on the left of CB
            CBleft, _ = separate_points_by_line(C, B, pointcloud)

            #Recursively process each set
            query_line(ACleft, A, C)
            query_line(CBleft, C, B)

            return;

        def separate_points_by_line(A, B, P):
            val = (P[:, 1] - A[1]) * (B[0] - A[0]) - (B[1] - A[1]) * (P[:, 0] - A[0])
            # Πάλι εξωτερικό γινόμενο αλλά με ορίζουσες κατευθείαν (Κουλουρίδης τύπος)

            left_points = P[val > 0]  # Βρίσκονται στα αριστερά
            right_points = P[val < 0] # Βρίσκονται στα δεξιά

            return left_points, right_points;
                    
        #Finding extreme points
        A, B = self.pcd.points[np.argmin(self.pcd.points[:, 0])], self.pcd.points[np.argmax(self.pcd.points[:, 0])]

        #list to keep track of convex hull points
        chull = []

        #extreme points necessarily belong to the convex hull
        chull.append(A)
        chull.append(B)

        #splitting the pointcloud along the line into 2 sets
        P1, P2 = separate_points_by_line(A, B, self.pcd.points)

        #recusrively processing each point set
        query_line(P1, A, B)
        query_line(P2, B, A)

        self.poly = Polygon2D(chull, reorderIfNecessary=True, color = Color.ORANGE)
        self.addShape(self.poly, "ch")

        return;

# ----- Ερώτημα 4ο -----

    def CH_jarvis_march(self): # aka the Gift-Wrapping algorithm
        if len(self.pcd) < 3:
            raise ValueError("No convex hull can be defined with less than 3 points!")

        def orientation(p, q, r):
            return np.cross(q - p, r - q);

        def checkSamePoint(p1, p2):
            statement = False
            if p1[0] == p2[0] and p1[1] == p2[1]:
                statement = True
            
            return statement;

        base_p_idx = np.argmin(self.pcd.points[:, 0])
        base_p = self.pcd.points[base_p_idx] # Leftmost point
        
        extreme_points = [base_p]
        current_p = base_p
        while True:
            candidate = None
            for p in self.pcd.points:
                if not checkSamePoint(p, current_p):
                    candidate = p
                    break;
            
            for p in self.pcd.points:
                if checkSamePoint(p, current_p):
                    continue;
                
                # Για απλότητα, παραλείπω τα συνευθειακά σημεία!
                if orientation(current_p, candidate, p) >= 0:
                    candidate = p

            current_p = candidate
            if checkSamePoint(current_p, base_p):
                break;

            extreme_points.append(current_p)

        self.poly = Polygon2D(extreme_points, color = Color.MAGENTA)
        self.addShape(self.poly, "ch")

        return;

# ----- Ερώτημα 6ο -----

    def is_point_inside_polygon(self, point: "Point2D"):
        temp = 0
        results = np.array([line.isOnRight(point) for line in self.poly.convex_hull_edges])
        if np.all(results) or np.all(~results):
            temp = 1

        return temp;

    def exercise6b_solution2(self, point: "Point2D", tempColor: Color = Color.BLUE):
        # Χρειάζεται να ισχύει reorderIfNecessary=True στο Polygon2D obj ώστε να δουλέψει σωστά!
        try: # Cheatinggg
            self.removeShape("max_angle_line-0")
            self.removeShape("max_angle_line-1")
        except:
            pass

        # https://docs.python.org/3/library/functions.html#hasattr
        my_function_obj = self.__class__.exercise6b_solution2
        if not hasattr(my_function_obj, "original_color"): # Για να μην χάσω το αρχικό χρώμα!
            my_function_obj.original_color = self.poly.colors[0]

        color_matrix = np.zeros((self.poly.colors.shape[0], 1)) + my_function_obj.original_color
        is_right = np.array([e.isOnRight(point) for e in self.poly.convex_hull_edges])
        color_matrix[~is_right] = tempColor
        self.poly.colors = color_matrix

        if self.is_point_inside_polygon(point):
            self.updateShape("ch") # Επαναφορά χρωμάτων!
            return;

        # Πρόσθεσε τις γραμμές που συνδέουν το σημείο με τα άκρα του ορατού πολυγώνου
        temp = [np.where(is_right)[0], np.where(~is_right)[0]]
        for visible_indices in temp:
            first_edge = self.poly.convex_hull_edges[visible_indices[0]]
            last_edge = self.poly.convex_hull_edges[visible_indices[-1]]

            first_visible_vertex = (first_edge.x1, first_edge.y1)
            last_visible_vertex = (last_edge.x2, last_edge.y2)

            if first_visible_vertex != last_visible_vertex:
                break;

        line1 = Line2D(point, first_visible_vertex, color = tempColor)
        line2 = Line2D(point, last_visible_vertex, color = tempColor)
        self.addShape(line1, "max_angle_line-0")
        self.addShape(line2, "max_angle_line-1")

        self.updateShape("ch")
        
        return;

    def is_point_inside_polygon_intersection(self, point: "Point2D"):
        for v in self.poly.convex_hull_vertices:
            for e in self.poly.convex_hull_edges:
                if (v.x == e.x1 and v.y == e.y1) or (v.x == e.x2 and v.y == e.y2):
                    continue;

                p1 = Point2D((e.x1, e.y1))
                p2 = Point2D((e.x2, e.y2))
                if intersect(p1, p2, v, self.my_mouse_pos):
                    return 0;

        return 1;

    def exercise6b(self, point: "Point2D", tempColor: Color = Color.BLUE):
        try: # Cheatinggg
            self.removeShape("max_angle_line-0")
            self.removeShape("max_angle_line-1")

            for i in range(len(self.poly.convex_hull_vertices)):
                self.removeShape(f"new_edge_line-{i}")
        except:
            pass

        if self.is_point_inside_polygon_intersection(point):
            return;

        not_visible = []
        for v in self.poly.convex_hull_vertices:
            for e in self.poly.convex_hull_edges:
                if (v.x == e.x1 and v.y == e.y1) or (v.x == e.x2 and v.y == e.y2):
                    continue;

                p1 = Point2D((e.x1, e.y1))
                p2 = Point2D((e.x2, e.y2))
                if intersect(p1, p2, v, self.my_mouse_pos):
                    not_visible.append((v.x, v.y))

        visible = [(v.x, v.y) for v in self.poly.convex_hull_vertices if (v.x, v.y) not in not_visible]
        
        temp_idx = 0
        for e in self.poly.convex_hull_edges:
            if ((e.x1, e.y1) in visible) and ((e.x2, e.y2) in visible):
                red_line = Line2D((e.x1, e.y1), (e.x2, e.y2), color = tempColor)
                self.addShape(red_line, f"new_edge_line-{temp_idx}")
                temp_idx += 1

        temp = find_max_angle_pair((point.x, point.y), visible)
        for i in range(len(temp)):
            self.addShape(Line2D(point, temp[i], color = tempColor), f"max_angle_line-{i}")

        return;

    def on_mouse_press(self, x, y, button, modifiers):
        self.my_mouse_pos.x = x
        self.my_mouse_pos.y = y
        self.my_mouse_pos.color = [Color.BLUE, Color.MAGENTA][self.is_point_inside_polygon(self.my_mouse_pos)]
        self.updateShape("mouse")

        self.exercise6b_solution2(self.my_mouse_pos)

        return;

    def on_mouse_release(self, x, y, button, modifiers):
        self.my_mouse_pos.x = x
        self.my_mouse_pos.y = y
        self.my_mouse_pos.color = [Color.WHITE, Color.BLACK][self.is_point_inside_polygon(self.my_mouse_pos)]
        self.updateShape("mouse")

        return;

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        self.on_mouse_press(x, y, buttons, modifiers)

        return;

# https://stackoverflow.com/questions/3838329/how-can-i-check-if-two-segments-intersect
def ccw(A,B,C):
    return (C.y-A.y) * (B.x-A.x) > (B.y-A.y) * (C.x-A.x);

# Return true if line segments AB and CD intersect
def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D);
#######################################################################################

def angle_between(v1, v2):
    dot = np.einsum('ij,ij->i', v1, v2) # Vectorized way to compute dot
                                        # products between pairs of vectors!
    norms = np.linalg.norm(v1, axis = 1) * np.linalg.norm(v2, axis = 1)
    cos_theta = np.clip(dot / norms, -1., 1.)

    return np.arccos(cos_theta);

def find_max_angle_pair(base: tuple, points: tuple):
    length = len(points)
    if length < 2:
        raise ValueError("At least 2 points are required!")

    base = np.asarray(base)
    points = np.asarray(points)
    vectors = points - base

    idx_pairs = np.array(list(combinations(range(length), 2)))
    v1 = vectors[idx_pairs[:, 0]]
    v2 = vectors[idx_pairs[:, 1]]

    angles = angle_between(v1, v2)
    max_idx = np.argmax(angles)
    i1, i2 = idx_pairs[max_idx]
    best_pair = (tuple(points[i1]), tuple(points[i2]))

    return best_pair;

if __name__ == "__main__":
    app = Lab2()
    app.mainLoop()
