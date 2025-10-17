from vvrpywork.constants import Key, Mouse, Color
from vvrpywork.scene import Scene2D, Scene3D, get_rotation_matrix, world_space
from vvrpywork.shapes import (
    Point2D, Line2D, Triangle2D, Circle2D, Rectangle2D,
    PointSet2D, LineSet2D, Polygon2D, Label2D,
    Point3D, Line3D, Arrow3D, Sphere3D, Cuboid3D, Cuboid3DGeneralized,
    PointSet3D, LineSet3D, Mesh3D, Label3D
)

from math import atan2 # for polar coordinates
import numpy as np

WIDTH = 800
HEIGHT = 800

class Lab1_2D(Scene2D):
    def __init__(self):
        super().__init__(WIDTH, HEIGHT, "Lab1")
        self.reset()
        self.task1()

        return;

    def reset(self):
        # create a point
        p = Point2D((0, 0), size = 4, color = Color.CYAN)
        
        # display the point
        self.addShape(p)

        # create and display a second point
        q = Point2D((0.2, 0.4), size = 2, color = Color.MAGENTA)
        self.addShape(q)

        # connect the points with a line
        pq = Line2D(p, q)
        self.addShape(pq)

        # calculate the distance between the points
        dist = self.calcDistance(p, q)
        print(f"dist = {dist:.3f}")

        # make a rectangle
        rect = Rectangle2D((-0.5, -0.5),
                           (+0.5, +0.5))
        self.addShape(rect)

        # generate a random point cloud
        self.my_point_cloud = PointSet2D()
        self.my_point_cloud.createRandom(rect, 10, seed = "compgeo", color = Color.RED)
        self.my_point_cloud.size = 1.5
        self.addShape(self.my_point_cloud)

        # polygon creation
        self.polygon_creation()

        # add a point to visualize mouse clicks
        self.my_mouse_pos = Point2D((0, 0), color=(0, 0, 0, 0))
        self.addShape(self.my_mouse_pos, "mouse")
        # mouse => Το όνομα του αντικειμένου που δημιουργείται.
        # Δες self.updateShape("mouse") στη μέθοδο on_mouse_press!

        return;

    def calcDistance(self, p1:Point2D, p2:Point2D) -> float:
        # calculate the distance between p1 and p2
        dx = p1.x - p2.x
        dy = p1.y - p2.y
        distance = (dx**2 + dy**2)**0.5

        return distance;
    
    def polygon_creation(self):
        ls = LineSet2D()
        num_pts = len(self.my_point_cloud.points)

        # connect all points with lines
        for i in range(num_pts):
            p1 = self.my_point_cloud[i]
            for j in range(num_pts):
                p2 = self.my_point_cloud[j]
                line = Line2D(p1, p2)
                ls.add(line)
                # Το εξωτερικό περίβλημα που εμφανίζεται,
                # ονομάζεται κυρτό περίβλημα!
        
        #self.addShape(ls)

        # create the polygon
        poly = Polygon2D(self.my_point_cloud, reorderIfNecessary = True, color = Color.YELLOW)
        # reorderIfNecessary = True => Μη επικαλυπτόμενο!

        self.addShape(poly)

        return;

    def on_mouse_press(self, x, y, button, modifiers):
        self.my_mouse_pos.x = x
        self.my_mouse_pos.y = y
        self.my_mouse_pos.color = Color.MAGENTA
        self.updateShape("mouse")

        return;
    
    def on_mouse_release(self, x, y, button, modifiers):
        self.my_mouse_pos.x = x
        self.my_mouse_pos.y = y
        self.my_mouse_pos.color = Color.WHITE
        self.updateShape("mouse")

        return;

    def task1(self):
        # a. compute the center of mass of the point cloud
        center_mass = np.mean(self.my_point_cloud.points, axis = 0)
        print(f"Κέντρο μάζας του νέφους σημείων: [x y] = {center_mass}")
        # https://numpy.org/doc/2.2/reference/generated/numpy.mean.html

        #self.addShape(Point2D(center_mass, size = 4, color = Color.GREEN))

        # b. shift the whole point cloud so that its center of mass becomes (0,0)
        "Αφαιρώ το κέντρο μάζας από τις συντεταγμένες των σημείων:"
        my_cloud_coords_centered = self.my_point_cloud.points - center_mass

        '''
        self.my_point_cloud.colors = np.ones((my_cloud_coords_centered.size, 4)) * Color.GREEN
        self.my_point_cloud.points = my_cloud_coords_centered
        self.addShape(self.my_point_cloud) # Τελικά, τα έχω σχεδιάσει παρακάτω
        self.addShape(Point2D(center_mass - center_mass, size = 4, color = Color.RED))
        '''

        # c. convert all points to polar coordinates system
        "r**2 = x**2 + y**2, tan(θ) = y/x"
        x = my_cloud_coords_centered[:, 0]
        y = my_cloud_coords_centered[:, 1]
        
        r = (x**2 + y**2)**0.5
        r = np.reshape(r, (-1, 1)) # Μετατροπή του array σε column vector
        theta = np.array([atan2(y[i], x[i]) for i in range(len(x))])
        theta = np.reshape(theta, (-1, 1))
        
        polar_coords = np.hstack((r, theta))
        # https://numpy.org/doc/stable/reference/generated/numpy.hstack.html
        
        #print(polar_coords)

        # d. sort points by angle
        "Κατά αύξουσα σειρά με βάση την πολική γωνία:"
        my_cloud_coords_sorted = my_cloud_coords_centered[polar_coords[:, 1].argsort()]

        # e. create the polygon using the sorted points (leave reorderIfNecessary=False)
        poly_ls = LineSet2D()
        num_pts = len(my_cloud_coords_sorted)
        for i in range(-1, num_pts - 1):
            p1 = Point2D(my_cloud_coords_sorted[i],
                         size = 1.5, color = Color.DARKGREEN) # Point2D, so I can draw it!
            p2 = my_cloud_coords_sorted[i + 1]
            line = Line2D(p1, p2, color = Color.ORANGE)
            poly_ls.add(line)
            self.addShape(p1, f"point_cloud_cs{i}")

        self.addShape(poly_ls)

        return;

#######################################################################################################

class Lab1_3D(Scene3D):
    def __init__(self):
        super().__init__(WIDTH, HEIGHT, "Lab1")
        self.reset()
        self.task2()

        return;

    def reset(self):
        m = Mesh3D.create_bunny()
        # self.addShape(m)
        self.pcd = PointSet3D(m.vertices, color=Color.BLUE)
        self.addShape(self.pcd, "pcd")

        return;

    def task2(self):
        # a. apply anisotropic scaling
        scale = np.array([40, 10, 30])
        new_points = self.anisotropic_scale(self.pcd.points, scale)
        self.pcd.points = new_points
        self.updateShape("pcd")

        # b. shift the point cloud so that its center of mass becomes (0,0)
        new_points = self.recenter(self.pcd.points)
        self.pcd.points = new_points
        self.updateShape("pcd")

        # c. normalize the pointcloud to be inside the unit sphere
        unit_sphere = Sphere3D((0, 0, 0), 1, color = (1, 0, 0))
        self.addShape(unit_sphere, "unit_sphere")

        new_points = self.unit_sphere_normalization(self.pcd.points)
        self.pcd.points = new_points
        self.updateShape("pcd")

        # d. change each point's color according to its distance from (0, 0, 0)
        new_colors = self.distance_colors(self.pcd.points)
        self.pcd.colors = new_colors
        self.removeShape("unit_sphere")
        self.updateShape("pcd")

        # e. delete points not passing a certain threshold
        THRESHOLD = 0.5
        threshold_sphere = Sphere3D((0, 0, 0), THRESHOLD)
        self.addShape(threshold_sphere, "threshold_sphere")

        idx = self.distance_threshold(self.pcd.points, THRESHOLD, "less")
        # Use boolean array for indexing
        self.pcd.points = self.pcd.points[idx]
        self.pcd.colors = self.pcd.colors[idx]
        self.updateShape("pcd")

        return;

    def anisotropic_scale(self, points:np.ndarray, scale:np.ndarray) -> np.ndarray:
        '''Applies anisotropic scaling.

        Args:
            points: The given point cloud with N points. Shape: (N, 3)
            scale: Contains the scaling factor for the x, y, z axes respectively. Shape: (3,)

        Returns:
            A numpy array containing the new points. Shape: (N, 3)
        '''
        # https://numpy.org/doc/stable/user/basics.broadcasting.html
        assert scale.shape == (3,)

        return points * scale;
    
    def recenter(self, points:np.ndarray) -> np.ndarray:
        '''Centers the point cloud by subtracting the center of mass.

        Args:
            points: The given point cloud with N points. Shape: (N, 3)

        Returns:
            A numpy array containing the new points. Shape: (N, 3)
        '''
        center_mass = np.mean(points, axis = 0)
        centered_points = points - center_mass

        return centered_points;
    
    def unit_sphere_normalization(self, points:np.ndarray) -> np.ndarray:
        '''Applies unit sphere normalization.

        Args:
            points: The given point cloud with N points. Shape: (N, 3)

        Returns:
            A numpy array containing the new points. Shape: (N, 3)
        '''
        # Spherical coordinate system
        r = np.sqrt(np.sum(np.square(points), axis = 1))
        r_max = np.max(r) # Βρες το max radial distance!
        r_normalized = np.divide(points, r_max)

        return r_normalized;
    
    def distance_colors(self, points:np.ndarray) -> np.ndarray:
        '''Return an array of colors based on the distance of each point form the origin.

        Args:
            points: The given point cloud with N points. Shape: (N, 3)

        Returns:
            A numpy array containing the new colors of the points. Shape: (N, 3)
        '''
        # Spherical coordinate system (radial distance), but normalized {0 ≤ r ≤ 1}
        r = np.linalg.norm(points, axis = 1)
        # https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html

        color = np.zeros((r.size, 3)) + r.reshape((-1, 1)) # 3 columns because R G B!
        color_value = color * (np.array([220, 20, 60]) / 255) # crimson
        # https://www.rapidtables.com/web/color/RGB_Color.html

        return color_value;
    
    def distance_threshold(self, points:np.ndarray, threshold:float, mode:str="greater") -> np.ndarray:
        '''Finds which points pass a certain threshold.
        
        Args:
            points: The given point cloud with N points. Shape: (N, 3)
            threshold: The threshold to switch between keeping and discarding points.
            mode: Either `greater` or `less`. Whether to keep points *greater* than or
                *less* than the threshold.

        Returns:
            A boolean numpy array containing `True` if the point at that index passes
                the threshold and `False` if it doesn't. Shape: (N,)
        '''
        # https://numpy.org/doc/stable/reference/generated/numpy.greater.html
        assert mode in ("greater", "less"), "Mode can only be 'greater' or 'less'"
        
        r = np.linalg.norm(points, axis = 1)
        result = r > threshold
        if mode == "less":
            result = r < threshold
        
        return result;

if __name__ == "__main__":
    app = Lab1_2D() # CHANGE TO Lab1_3D FOR TASK 2
    app.mainLoop()