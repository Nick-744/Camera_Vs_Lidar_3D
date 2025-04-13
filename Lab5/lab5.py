from vvrpywork.constants import Key, Color
from vvrpywork.scene import Scene3D, Scene3D_
from vvrpywork.shapes import Point3D, Sphere3D, Cuboid3D, PointSet3D

import heapq
import numpy as np

TASK = 5 # Change this to 1 to run the task

WIDTH = 1000
HEIGHT = 800

COLORS = [Color.RED, Color.GREEN, Color.BLUE, Color.YELLOW, Color.ORANGE, Color.MAGENTA, Color.YELLOWGREEN, Color.CYAN]

class Lab5(Scene3D_):
    def __init__(self):
        super().__init__(WIDTH, HEIGHT, "Lab5", output=True, n_sliders=4)
        self.reset()

        return;

    def reset(self):
        self.set_slider_value(0, 0.5) # Slider 0 - Point Speed
        self.p = Point3D((-1, 0, 0))
        self.addShape(self.p, "p")
        self.paused = True

        self.kd_depth = 0
        self.earlier_pcd = PointSet3D(color=Color.WHITE)
        self.addShape(self.earlier_pcd, "earlier_pcd")

        self.bounds = Cuboid3D((-0.9, -0.6, -0.3), (0.9, 0.6, 0.3), color=Color.WHITE)
        self.addShape(self.bounds)
        self.pts = PointSet3D()
        self.set_slider_value(1, 0.2) # Slider 1 - Number of Points

        self.pts_in_sphere = PointSet3D(size=1.6, color=Color.DARKGREEN)
        self.addShape(self.pts_in_sphere, "pts_in_sphere")

        self.sphere = Sphere3D(self.p, resolution=20, color=Color.WHITE, filled=False)
        self.addShape(self.sphere, "sphere")
        self.set_slider_value(2, 0) # Slider 2 - Sphere Radius

        self.nn = PointSet3D(size=2, color=Color.BLACK)
        self.addShape(self.nn, "nn")

        if TASK >= 4:
            if hasattr(self.root, "pivot"):
                nn = KdNode.nearestNeighbor(self.p, self.root)
                self.nn.clear()
                self.nn.add(Point3D(nn.pivot))
                self.updateShape("nn")

        self.knn = PointSet3D(size=1.4, color=Color.DARKORANGE)
        self.addShape(self.knn, "knn")
        self.set_slider_value(3, 0) # Slider 3 - K : Number of Nearest Neighbors

        return;

    def on_key_press(self, symbol, modifiers):
        if symbol == Key.SPACE:
            self.paused = not self.paused
        if TASK >= 2:
            if symbol == Key.UP:
                if self.kd_depth <= self.max_depth:
                    self.erase_kd_tree()
                    self.kd_depth += 1
                    self.redraw_kd_tree()
            if symbol == Key.DOWN:
                if self.kd_depth > 0:
                    self.erase_kd_tree()
                    self.kd_depth -= 1
                    self.redraw_kd_tree()

        return;

    def on_slider_change(self, slider_id, value):
        if slider_id == 0:
            self.point_speed = value * 0.01

        if slider_id == 1:
            self.erase_kd_tree()
            self.kd_depth = 0
            self.pts.clear()
            self.pts.createRandom(self.bounds, int(value * 4999) + 1, "lab5")
            self.root = KdNode(self.pts.points, 0)
            if TASK >= 2:
                self.max_depth = KdNode.getMaxDepth(self.root)
            self.redraw_kd_tree()
            self.print(f"{int(value * 4999) + 1} points")

        if slider_id == 2:
            self.sphere.radius = value * 0.8
            self.updateShape("sphere")

            if TASK >=3:
                if hasattr(self.root, "pivot"):
                    pts = KdNode.inSphere(self.sphere, self.root)
                    self.pts_in_sphere.clear()
                    self.pts_in_sphere.points = pts
                    self.pts_in_sphere.colors = np.tile(Color.DARKGREEN, len(pts)).reshape(-1, 4)
                    self.updateShape("pts_in_sphere")

        if slider_id == 3:
            self.k = int((value * 100) + 0.5)

            if TASK >= 5:
                if hasattr(self.root, "pivot"):
                    nodes = KdNode.nearestK(self.p, self.root, self.k)
                    self.knn.clear()
                    pts = tuple(n.pivot for n in nodes)
                    self.knn.points = np.array(pts)
                    self.knn.colors = np.tile(Color.DARKORANGE, len(pts)).reshape(-1, 4)
                    self.updateShape("knn")
                    self.print(f"{self.k} NN [Task 5]")

        return;

    def on_idle(self):
        if not self.paused:
            self.p.x += self.point_speed # Move the center of the sphere
            if self.p.x > 1:
                self.p.x = -1
            self.updateShape("p", True)

            self.sphere.x, self.sphere.y, self.sphere.z = self.p.x, self.p.y, self.p.z
            self.updateShape("sphere")

            if TASK >= 3:
                pts = KdNode.inSphere(self.sphere, self.root)
                self.pts_in_sphere.clear()
                self.pts_in_sphere.points = pts
                self.pts_in_sphere.colors = np.tile(Color.DARKGREEN, len(pts)).reshape(-1, 4)
                self.updateShape("pts_in_sphere")

            if TASK >= 4:
                nn = KdNode.nearestNeighbor(self.p, self.root)
                self.nn.clear()
                self.nn.add(Point3D(nn.pivot))
                self.updateShape("nn")

            if TASK >= 5:
                nodes = KdNode.nearestK(self.p, self.root, self.k)
                self.knn.clear()
                pts = tuple(n.pivot for n in nodes)
                self.knn.points = np.array(pts)
                self.knn.colors = np.tile(Color.DARKORANGE, len(pts)).reshape(-1, 4)
                self.updateShape("knn")

            return True;
        return False;
    
    def erase_kd_tree(self):
        for i in range(2 ** (self.kd_depth)):
            self.removeShape("kd" + str(i))
            self.removeShape("aabb" + str(i))
        self.earlier_pcd.clear()
        self.updateShape("earlier_pcd")

        return;
    
    def redraw_kd_tree(self):
        if TASK >= 2: 
            # Select the nodes at the selected depth
            nodes = KdNode.getNodesAtDepth(self.root, self.kd_depth) 
            # Assign a different colour to the points in each node
            if nodes is not None:
                for (i, node) in enumerate(nodes):
                    pcd = PointSet3D(KdNode.getNodesBelow(node), color=COLORS[i % len(COLORS)])
                    pcd.add(Point3D(node.pivot, color=COLORS[i % len(COLORS)]))
                    self.addShape(pcd, "kd" + str(i))
                    if len(pcd) > 1:
                        aabb = pcd.getAABB()
                        aabb.color = COLORS[i % len(COLORS)]
                        self.addShape(aabb, "aabb" + str(i)) # Axis-Aligned Bounding Box
                
                # Make all pivot points to the current depth white
                self.earlier_pcd.clear()
                for i in range(self.kd_depth):
                    for node in KdNode.getNodesAtDepth(self.root, i):
                        self.earlier_pcd.add(Point3D(node.pivot, color=Color.WHITE))
                self.updateShape("earlier_pcd")
        else:
            # For TASK 1, just visualize all points with white color
            pcd = PointSet3D(self.pts.points, color=Color.WHITE)
            self.addShape(pcd, "kd0")

        return;

class KdNode:
    def __init__(self, pts: np.ndarray, depth: int):
        if len(pts) < 1:
            return;
        
        # Select the axis to do the split
        axis = depth % 3

        # Find the median of the points along the selected axis (pivot)
        indices = np.argsort(pts[:, axis])
        # Select all rows (:), and only the column at index axis!
        sorted_pts = pts[indices]

        num_points = len(sorted_pts)
        median_index = num_points // 2

        self.pivot = sorted_pts[median_index]
        self.depth = depth

        left_points = sorted_pts[:median_index]
        num_left_points = len(left_points)
        self.left_child = KdNode(left_points, depth + 1) if num_left_points > 0 else None

        right_points = sorted_pts[median_index + 1:] # +1, για να παραλείψουμε το pivot!
        num_right_points = len(right_points)
        self.right_child = KdNode(right_points, depth + 1) if num_right_points > 0 else None

        return;

    def __repr__(self) -> str:
        return f"k-d node @ {self.pivot}, depth = {self.depth}";
    
    @staticmethod
    def getMaxDepth(node: 'KdNode') -> int:
        if (node.left_child is None) and (node.right_child is None):
            return node.depth;
        elif node.left_child is None:
            return KdNode.getMaxDepth(node.right_child);
        elif node.right_child is None:
            return KdNode.getMaxDepth(node.left_child);
        else:
            return max(KdNode.getMaxDepth(node.left_child), KdNode.getMaxDepth(node.right_child));
    
    @staticmethod
    def getNodesBelow(node: 'KdNode') -> np.ndarray:

        def _getNodesBelow(node: 'KdNode', pts: list):
            if node.left_child is not None:
                pts.append(node.left_child.pivot)
                pts = _getNodesBelow(node.left_child, pts)
            
            if node.right_child is not None:
                pts.append(node.right_child.pivot)
                pts = _getNodesBelow(node.right_child, pts)

            return pts;

        pts = _getNodesBelow(node, [])

        return np.array(pts);
    
    @staticmethod
    def getNodesAtDepth(node: 'KdNode', depth: int) -> list:

        def _getNodesAtDepth(node: 'KdNode', depth: int, pts: list):
            if node.depth == depth: # Δεν θα το ξεπεράσει ποτέ το βάθος, λόγω της getMaxDepth!
                pts.append(node)
                return pts;

            if node.left_child is not None:
                pts = _getNodesAtDepth(node.left_child, depth, pts)
            
            if node.right_child is not None:
                pts = _getNodesAtDepth(node.right_child, depth, pts)

            return pts;

        pts = _getNodesAtDepth(node, depth, [])

        return pts;
    
    @staticmethod
    def inSphere(sphere: Sphere3D, node: 'KdNode') -> np.ndarray:
        
        def _inSphere(sphere: Sphere3D, node: 'KdNode', pts: list):
            # Center of the sphere, node.pivot
            pivot = node.pivot
            d_sq = (sphere.x - pivot[0]) * (sphere.x - pivot[0]) + \
                   (sphere.y - pivot[1]) * (sphere.y - pivot[1]) + \
                   (sphere.z - pivot[2]) * (sphere.z - pivot[2])
            
            if d_sq <= (sphere.radius * sphere.radius):
                pts.append(pivot)
            
            axis = node.depth % 3
            d_pivot = (pivot[0] - sphere.x, pivot[1] - sphere.y, pivot[2] - sphere.z)[axis]

            if d_pivot <= 0:
                check_first = node.right_child
                check_second = node.left_child
            else:
                check_first = node.left_child
                check_second = node.right_child

            if check_first is not None:
                pts = _inSphere(sphere, check_first, pts)
            
            if abs(d_pivot) <= sphere.radius:
                if check_second is not None:
                    pts = _inSphere(sphere, check_second, pts)

            return pts;
    
        pts = _inSphere(sphere, node, [])
        
        return np.array(pts);
    
    @staticmethod
    def nearestNeighbor(test_pt: Point3D, node: 'KdNode') -> 'KdNode':

        def _nearestNeighbor(test_pt:Point3D, node:'KdNode', nn:'KdNode', best_dist:float):
            # Check the distance between the test point and the pivot of the current node
            pivot = node.pivot
            d_sq = (pivot[0] - test_pt.x) * (pivot[0] - test_pt.x) + \
                   (pivot[1] - test_pt.y) * (pivot[1] - test_pt.y) + \
                   (pivot[2] - test_pt.z) * (pivot[2] - test_pt.z)

            if d_sq < best_dist: # Δεν θέλει τώρα πια τετράγωνο!
                best_dist = d_sq # Αφού το best_dist είναι ήδη στο τετράγωνο!
                nn = node
            
            # Check which side of the split plane the test point is on
            axis = node.depth % 3
            d_pivot = (pivot[0] - test_pt.x, pivot[1] - test_pt.y, pivot[2] - test_pt.z)[axis]

            if d_pivot <= 0:
                check_first = node.right_child
                check_second = node.left_child
            else:
                check_first = node.left_child
                check_second = node.right_child
            
            # Check the 1st child node
            if check_first is not None:
                (nn, best_dist) = _nearestNeighbor(test_pt, check_first, nn, best_dist)

            # Check the 2nd child node if necessary
            if (d_pivot * d_pivot) <= best_dist:
                if check_second is not None:
                    (nn, best_dist) = _nearestNeighbor(test_pt, check_second, nn, best_dist)
            
            return (nn, best_dist);

        (nn, _) = _nearestNeighbor(test_pt, node, None, np.inf)

        return nn;
    
    @staticmethod
    def nearestK(test_pt: Point3D, node: 'KdNode', k: int) -> list['KdNode']:

        def _nearestK(test_pt:Point3D, node:'KdNode', k:int, heap:list[tuple[int,'KdNode']], d_threshold:float):
            pivot = node.pivot
            d_sq = (test_pt.x - pivot[0]) * (test_pt.x - pivot[0]) + \
                   (test_pt.y - pivot[1]) * (test_pt.y - pivot[1]) + \
                   (test_pt.z - pivot[2]) * (test_pt.z - pivot[2])
            
            if len(heap) < k: # Δεν έχουμε γεμήσει ακόμα το heap!
                heapq.heappush(heap, (-d_sq, node)) # -d_sq, γιατί υλοποιεί min heap,
                                                    # ενώ θέλουμε να έχουμε max heap!
                d_threshold = -heap[0][0]
            else:
                if d_sq < d_threshold:
                    heapq.heapreplace(heap, (-d_sq, node))
                    d_threshold = -heap[0][0]
        
            # Check which side of the split plane the test point is on
            axis = node.depth % 3
            d_pivot = (pivot[0] - test_pt.x, pivot[1] - test_pt.y, pivot[2] - test_pt.z)[axis]

            if d_pivot <= 0:
                check_first = node.right_child
                check_second = node.left_child
            else:
                check_first = node.left_child
                check_second = node.right_child

            # Check the 1st child node
            if check_first is not None:
                (heap, d_threshold) = _nearestK(test_pt, check_first, k, heap, d_threshold)

            # Check the 2nd child node if necessary
            if (d_pivot * d_pivot) <= d_threshold:
                if check_second is not None:
                    (heap, d_threshold) = _nearestK(test_pt, check_second, k, heap, d_threshold)

            return (heap, d_threshold);
        
        if k == 0:
            return [];

        (heap, _) = _nearestK(test_pt, node, k, [], np.inf)

        return [n for (_, n) in heap];

# ******************************** Homework - 6 ******************************** #
''' Υλοποιήστε τα ερωτήματα 3-5 χωρίς να χρησιμοποιήσετε το kd-tree. Συγκρίνετε το
χρόνο εκτέλεσης των συναρτήσεων και παρουσιάστε τα αποτελέσματά σας για
διάφορα πλήθη σημείων. Εξηγείστε τι παρατηρείτε. '''

def inSphere_brute_force(sphere: Sphere3D, points: np.ndarray) -> np.ndarray:
    temp = []
    radius_sq = sphere.radius * sphere.radius

    for p in points:
        d_sq = (sphere.x - p[0]) * (sphere.x - p[0]) + \
               (sphere.y - p[1]) * (sphere.y - p[1]) + \
               (sphere.z - p[2]) * (sphere.z - p[2])
        if d_sq <= radius_sq:
            temp.append(p)
    
    return np.array(temp);

def nearest_helper(test_pt: Point3D, points: np.ndarray) -> np.ndarray:
    ''' Γυρνά την λίστα [NumPy] με τις αποστάσεις των σημείων από το test_pt '''

    temp = [
        (p[0] - test_pt.x) * (p[0] - test_pt.x) + \
        (p[1] - test_pt.y) * (p[1] - test_pt.y) + \
        (p[2] - test_pt.z) * (p[2] - test_pt.z) for p in points
    ]

    return np.array(temp);

def nearestNeighbor_brute_force(test_pt: Point3D, points: np.ndarray) -> Point3D:
    distances = nearest_helper(test_pt, points)
    temp = np.argsort(distances)
    
    return points[temp[0]];

def nearestK_brute_force(test_pt: Point3D, points: np.ndarray, k: int) -> list[Point3D]:
    distances = nearest_helper(test_pt, points)
    temp = np.argsort(distances)

    return points[temp[:k]];

from time import time
import matplotlib.pyplot as plt

def testFunctionSpeed(func, points_len, *args, printVariable = True):
    start = time()

    temp = func(*args)

    stop = time()
    executionTime = stop - start

    if printVariable:
        print(f'Χρόνος εκτέλεσης: {executionTime:.6f} δευτερόλεπτα', end = ' | ')
        print(f'Αριθμός σημείων: {points_len: 6}', end = ' | ')
        print(f'Συνάρτηση: {func.__name__}')
        
    return (temp, func.__name__, points_len, executionTime);

def question6():
    foos = [
        [inSphere_brute_force,        KdNode.inSphere],
        [nearestNeighbor_brute_force, KdNode.nearestNeighbor],
        [nearestK_brute_force,        KdNode.nearestK]
    ]

    labels = [
        ['inSphere [Brute Force]',        'inSphere [k-d tree]'],
        ['nearestNeighbor [Brute Force]', 'nearestNeighbor [k-d tree]'],
        ['nearestK [Brute Force]',        'nearestK [k-d tree]']
    ]

    num_points_ls = [1000, 5000, 10000, 20000]
    bounds = Cuboid3D((-1, -1, -1), (1, 1, 1))
    test_point = Point3D((0., 0., 0.))
    k = 10
    sphere = Sphere3D(test_point, radius = 0.5)
    
    for (i, (brute_force, kd_tree)) in enumerate(foos):
        brute_force_times = []
        kd_tree_times = []

        for num_points in num_points_ls:
            temp = PointSet3D()
            temp.createRandom(bounds, num_points)
            points = temp.points
            root = KdNode(points, 0)

            if i == 0:
                args_brute = (sphere, points)
                args_kd = (sphere, root)
            elif i == 1:
                args_brute = (test_point, points)
                args_kd = (test_point, root)
            else:
                args_brute = (test_point, points, k)
                args_kd = (test_point, root, k)

            (a, _, _, brute_time) = testFunctionSpeed(brute_force, num_points, *args_brute)
            (b, _, _, kd_time)    = testFunctionSpeed(kd_tree, num_points, *args_kd)

            # Έλεγχος αν βρήκαν τα ίδια!
            if isinstance(b, np.ndarray):
                assert(np.array_equal(np.sort(a, axis = None), np.sort(b, axis = None)))
            elif isinstance(b, KdNode):
                assert(np.array_equal(np.sort(a, axis = None), np.sort(b.pivot, axis = None)))
            else:
                b = [temp.pivot for temp in b]
                assert(np.array_equal(np.sort(a, axis = None), np.sort(b, axis = None)))
            
            brute_force_times.append(brute_time)
            kd_tree_times.append(kd_time)

        print() # Καλύτερη αισθητική

        # Plot results
        plt.figure()
        plt.plot(num_points_ls, brute_force_times, label = labels[i][0], marker = 'o')
        plt.plot(num_points_ls, kd_tree_times, label = labels[i][1], marker = 'x')
        plt.title(f"{labels[i][0]} Vs. {labels[i][1]}")
        plt.xlabel("Αριθμός Σημείων")
        plt.ylabel("Χρόνος Εκτέλεσης [sec]")
        plt.grid(True)
        plt.legend()
        plt.show()

    return;

# --- Choice --- #

import tkinter as tk

def choose_window():
    result = {'choice': None}

    def select(choice):
        result["choice"] = choice
        root.destroy()

        return;

    root = tk.Tk()
    root.title("Επίλεξε ")
    root.geometry("200x100")

    tk.Button(root, text = "BenchTest", command = lambda: select(1), width = 15).pack(pady = 10)
    tk.Button(root, text = "Lab - 5",   command = lambda: select(2), width = 15).pack()

    root.mainloop()

    return result["choice"];

if __name__ == "__main__":
    choice = choose_window()

    if choice == 1:
        print('BenchTest...')
        question6()
    elif choice == 2:
        print('Lab 5...')
        app = Lab5()
        app.mainLoop()
    else:
        print('Error')
