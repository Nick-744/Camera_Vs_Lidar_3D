from vvrpywork.constants import *
from vvrpywork.scene import *
from vvrpywork.shapes import *

from random import random

import numpy as np

WIDTH = 800
HEIGHT = 800

class Lab4(Scene3D):
    def __init__(self):
        super().__init__(WIDTH, HEIGHT, "Lab4")
        self.reset()

        return;

    def reset(self):
        self.model = Mesh3D.create_bunny(color=Color.GRAY)
        self.addShape(self.model, "model")
        self.model_w = LineSet3D.create_from_mesh(self.model)

        self.show_whole_model = True
        self.wireframe_shown = False

        self.planeY = 0
        self.planeAngle = 0.
        self.plane = np.array((0., 1., 0., 0.))
        self.plane_shape = Cuboid3DGeneralized(
            Cuboid3D(
                (-1, -0.0001, -1),
                (1, 0.0001, 1),
                color = (1, 0, 1, 0.5),
                filled = True
            )
        )
        self.addShape(self.plane_shape, "plane")

        # === TASK 1 ===
        centroid = self.task_1_find_centroid(self.model)
        print(centroid)

        # === TASK 2 ===
        self.task_2_translate_mesh(self.model, -centroid)
        self.model_w = LineSet3D.create_from_mesh(self.model)

        # === TASK 3 ===
        shpere_radius = 1
        self.task_3_scale_to_sphere(self.model, shpere_radius)
        self.model_w = LineSet3D.create_from_mesh(self.model)
        sphere = Sphere3D((0, 0, 0), shpere_radius, filled=False)
        # self.addShape(sphere)

        # === TASK 4 ===
        aabb = self.task_4_find_AABB(self.model)
        # self.addShape(aabb)

        # === TASK 5 ===
        principal_axes = self.task_5_PCA(self.model)
        first =  Arrow3D(
            (0, 0, 0),
            10 * principal_axes[0],
            color = Color.RED,
            cone_to_cylinder_ratio = 0.03
        )
        second = Arrow3D(
            (0, 0, 0),
            10 * principal_axes[1],
            color = Color.GREEN,
            cone_to_cylinder_ratio = 0.06
        )
        third =  Arrow3D((0, 0, 0), 10 * principal_axes[2], color = Color.BLUE)

        # self.addShape(first)
        # self.addShape(second)
        # self.addShape(third)

        # === TASK 6 ===
        intersections = self.task_6_plane_intersection(self.model, self.plane)
        self.model_i = Mesh3D(color=Color.CYAN)
        self.model_i.vertices = self.model.vertices
        self.model_i.triangles = intersections
        # self.addShape(self.model_i, "model_i")

        # === TASK 7 ===
        '''
        (top, bottom) = self.task_7_plane_split(self.model, self.plane)

        self.model_top = Mesh3D(color=Color.YELLOW)
        self.model_top.vertices = self.model.vertices + 0.2 * self.plane[:3]
        self.model_top.triangles = top
        self.model_top.vertex_normals = self.model.vertex_normals

        self.model_bottom = Mesh3D(color=Color.DARKRED)
        self.model_bottom.vertices = self.model.vertices
        self.model_bottom.triangles = bottom
        self.model_bottom.vertex_normals = self.model.vertex_normals

        self.show_whole_model = False
        '''

        # === TASK 8 ===
        precisely_cut_lines = self.task_8_exact_intersection(self.model.vertices[intersections], self.plane)
        self.addShape(precisely_cut_lines, 'task-8_lines')

        # === TASK 9 ===
        (   top_triangles,
            bottom_triangles,
            top_vertices,
            bottom_vertices
        ) = self.task_9_exact_split(self.model, self.plane)

        self.model_top = Mesh3D(color = Color.ORANGE)
        self.model_top.vertices = top_vertices + 0.005 * self.plane[:3]
        self.model_top.triangles = top_triangles

        self.model_bottom = Mesh3D(color = Color.WHITE)
        self.model_bottom.vertices = bottom_vertices
        self.model_bottom.triangles = bottom_triangles

        # Δείξε τα polygons του λαγού
        self.model_top_w = LineSet3D.create_from_mesh(self.model_top)
        self.model_bottom_w = LineSet3D.create_from_mesh(self.model_bottom)

        self.show_whole_model = False

        # -----

        if not self.show_whole_model:
            self.removeShape("model")
            if hasattr(self, "model_top"):
                self.addShape(self.model_top, "model_top")
            if hasattr(self, "model_bottom"):
                self.addShape(self.model_bottom, "model_bottom")

        return;

    def task_1_find_centroid(self, model:Mesh3D) -> np.ndarray:
        vertices = np.mean(model.vertices, axis = 0)

        return vertices;
    
    def task_2_translate_mesh(self, model:Mesh3D, translation:np.ndarray) -> None:
        model.vertices += translation
        self.updateShape("model")

        return;

    def task_3_scale_to_sphere(self, model:Mesh3D, radius:float) -> None:
        vertices = model.vertices
        distanceSq = (vertices * vertices).sum(axis = -1)
        max_dist = np.sqrt(np.max(distanceSq))
        # ή για όλα τα προηγούμενα, απλά:
        # max_dist = np.max(np.linalg.norm(model.vertices, axis = 1))

        scale = radius / max_dist
        model.vertices *= scale
        self.updateShape("model")

        return;

    def task_4_find_AABB(self, model:Mesh3D) -> Cuboid3D:
        vertices = model.vertices

        p1 = (
            np.min(vertices[:, 0]),
            np.min(vertices[:, 1]),
            np.min(vertices[:, 2])
        ) # ή = vertices.min(axis = 0)
        p2 = (
            np.max(vertices[:, 0]),
            np.max(vertices[:, 1]),
            np.max(vertices[:, 2])
        ) # ή = vertices.max(axis = 0)
        aabb = Cuboid3D(p1, p2) # Τα max και min, τα βρήκα προσημασμένα!
                                # Αυτά αποτελούν τις 2 άκρες του AABB!

        return aabb;
    
    def task_5_PCA(self, model:Mesh3D) -> np.ndarray:
        # Μπορεί να χρησιμοποιηθεί για Dimensionality Reduction,
        # αλλά εδώ δεν μας ενδιαφέρει αυτό!
        vertices = model.vertices

        covariance_matrix = np.cov(vertices, rowvar = False)
        # Διασπορά ως προς τις στήλες (δηλαδή τα x, y, z)
        # Note: Setting rowvar = False because each column represents a 
        #       variable, and each row a different observation (point)
        
        (eigenvalues, eigenvectors) = np.linalg.eig(covariance_matrix)
        
        sorted_indices = np.argsort(eigenvalues)[::-1] # Ταξινόμηση κατά φθίνουσα σειρά
                                                       # λόγω του [::-1]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]
        sorted_eigenvalues = eigenvalues[sorted_indices[:3]]

        principal_axes = sorted_eigenvectors[:, :3]
        principal_axes *= sorted_eigenvalues

        # Looney Tunes reference
        '''
        for i in range(len(sorted_eigenvalues)):
            print(f"Principal component {i + 1}")
            print(f"Eigenvector: {sorted_eigenvectors[:, i]}")
            print(f"Contibution to total variance: {sorted_eigenvalues[i] / np.sum(sorted_eigenvalues) * 100:.2f}%")
            print("-" * 40)

        (pc1, pc2, pc3) = sorted_eigenvectors[:, 0], sorted_eigenvectors[:, 1], sorted_eigenvectors[:, 2]
        projected_2D = np.dot(vertices, np.column_stack((pc1, pc2)))
        thin_component = np.dot(vertices, pc3) * 0.001
        flattend_vertices = np.column_stack((projected_2D, thin_component))
        model.vertices = flattend_vertices
        self.updateShape("model")
        '''

        return principal_axes.transpose();
    
    def task_6_plane_intersection(self, model:Mesh3D, plane:np.ndarray) -> np.ndarray:
        vertices = model.vertices
        triangles = model.triangles

        dist = np.dot(vertices, plane[:3]) - plane[3]
        intersecting = ~(
            np.all(dist[triangles] > 0, axis = 1) | np.all(dist[triangles] < 0, axis = 1)
        )
        
        return triangles[intersecting];
    
    def task_7_plane_split(self, model:Mesh3D, plane:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        vertices = model.vertices
        triangles = model.triangles

        dist = np.dot(vertices, plane[:3]) - plane[3]

        top = np.all(dist[triangles] > 0, axis = 1)
        bottom = np.all(dist[triangles] < 0, axis = 1)

        return triangles[top], triangles[bottom];
    
    def task_8_exact_intersection(self, intersecting_triangles: np.ndarray, plane: np.ndarray) -> 'LineSet3D':
        # intersecting_triangles = self.model.vertices[self.task_6_plane_intersection(self.model, self.plane)]
        
        # https://web.ma.utexas.edu/users/m408m/Display12-5-3.shtml
        (n, d) = (plane[:3], plane[3])
        # n * x = d {Plane Vector Equation}

        segments = [
            point for t in intersecting_triangles
            if (result := intersect_triangle_with_plane(t, n, d))
            for point in result # Αντί για append σε κάθε στοιχείο χωριστά!!!
        ] # https://docs.python.org/3/whatsnew/3.8.html {The Walrus Operator}

        lines = LineSet3D(segments, color = Color.RED)

        return lines;

    def task_9_exact_split(self, model: Mesh3D, plane: np.ndarray) -> tuple[list, list, list, list]:
        vertices = model.vertices
        triangles = model.triangles

        (n, d) = (plane[:3], plane[3])
        distances = np.dot(vertices, n) - d

        top = [] # Τα τρίγωνα του λαγού που είναι "πάνω" από το επίπεδο
        bottom = [] # Τρίγωνα λαγού "κάτω"!
        new_vertices_top = []
        new_vertices_bottom = []

        def add_top(v: np.ndarray):
            ''' Προσθέτει το σημείο v στο "πάνω" τμήμα του λαγού '''
            new_vertices_top.append(v)

            return len(new_vertices_top) - 1;

        def add_bottom(v: np.ndarray):
            ''' Προσθέτει το σημείο v στο "κάτω" τμήμα του λαγού '''
            new_vertices_bottom.append(v)

            return len(new_vertices_bottom) - 1;

        signs = distances[triangles] > 0
        for (t, sign) in zip(triangles, signs):
            (i0, i1, i2) = t

            if np.all(sign):
                idx = [add_top(vertices[i]) for i in t]
                top.append(idx)
                continue;
            elif not np.any(sign):
                idx = [add_bottom(vertices[i]) for i in t]
                bottom.append(idx)
                continue;

            triangle = [vertices[i0], vertices[i1], vertices[i2]]
            (p1, p2) = intersect_triangle_with_plane(np.array(triangle), n, d)
            (ip1_top, ip1_bot) = (add_top(p1), add_bottom(p1))
            (ip2_top, ip2_bot) = (add_top(p2), add_bottom(p2))

            if np.array_equal(sign, [True, True, False]):
                a = add_top(vertices[i0])
                b = add_top(vertices[i1])
                c = add_bottom(vertices[i2])
                top.append([a, ip2_top, b])
                top.append([ip1_top, ip2_top, b])
                bottom.append([c, ip1_bot, ip2_bot])
            elif np.array_equal(sign, [True, False, True]):
                a = add_top(vertices[i0])
                b = add_bottom(vertices[i1])
                c = add_top(vertices[i2])
                top.append([a, ip1_top, c])
                top.append([c, ip1_top, ip2_top])
                bottom.append([b, ip2_bot, ip1_bot])
            elif np.array_equal(sign, [False, True, True]):
                a = add_bottom(vertices[i0])
                b = add_top(vertices[i1])
                c = add_top(vertices[i2])
                top.append([b, c, ip2_top])
                top.append([b, ip1_top, ip2_top])
                bottom.append([a, ip2_bot, ip1_bot])
            elif np.array_equal(sign, [True, False, False]):
                a = add_top(vertices[i0])
                b = add_bottom(vertices[i1])
                c = add_bottom(vertices[i2])
                top.append([a, ip1_top, ip2_top])
                bottom.append([b, c, ip2_bot])
                bottom.append([b, ip2_bot, ip1_bot])
            elif np.array_equal(sign, [False, True, False]):
                a = add_bottom(vertices[i0])
                b = add_top(vertices[i1])
                c = add_bottom(vertices[i2])
                top.append([b, ip1_top, ip2_top])
                bottom.append([c, a, ip2_bot])
                bottom.append([a, ip2_bot, ip1_bot])
            elif np.array_equal(sign, [False, False, True]):
                a = add_bottom(vertices[i0])
                b = add_bottom(vertices[i1])
                c = add_top(vertices[i2])
                top.append([c, ip1_top, ip2_top])
                bottom.append([a, b, ip2_bot])
                bottom.append([b, ip2_bot, ip1_bot])

        return (top, bottom, new_vertices_top, new_vertices_bottom);

    def on_key_press(self, symbol, modifiers):
        if symbol == Key.W:
            if self.wireframe_shown:
                self.removeShape("model_w")
                self.wireframe_shown = False
            else:
                if self.show_whole_model and hasattr(self, "model_w"):
                    self.addShape(self.model_w, "model_w")
                self.wireframe_shown = True

        if symbol == Key.E: # Για Task 9!
            if self.wireframe_shown:
                if hasattr(self, "model_top_w"):
                    self.removeShape("model_top_w")
                if hasattr(self, "model_bottom_w"):
                    self.removeShape("model_bottom_w")
                self.wireframe_shown = False
            else:
                if hasattr(self, "model_top_w"):
                    self.addShape(self.model_top_w, "model_top_w")
                if hasattr(self, "model_bottom_w"):
                    self.addShape(self.model_bottom_w, "model_bottom_w")
                self.wireframe_shown = True

        if symbol in (Key.UP, Key.DOWN, Key.LEFT, Key.RIGHT):
            self.plane_shape.rotate(-self.planeAngle, (0, 0, 1))
            self.plane_shape.translate((0, -self.planeY, 0))
        
        if symbol == Key.UP:
            self.planeY += 0.01

        if symbol == Key.DOWN:
            self.planeY -= 0.01

        if symbol == Key.LEFT:
            self.planeAngle += 0.02

        if symbol == Key.RIGHT:
            self.planeAngle -= 0.02

        if symbol in (Key.UP, Key.DOWN, Key.LEFT, Key.RIGHT):
            self.plane_shape.translate((0, self.planeY, 0))
            self.plane_shape.rotate(self.planeAngle, (0, 0, 1))
            self.updateShape("plane")

            self.plane[:3] = get_rotation_matrix(self.planeAngle, (0, 0, 1)) @ (0, 1, 0)
            self.plane[3] = self.planeY

            # === TASK 6 ===
            intersections = self.task_6_plane_intersection(self.model, self.plane)
            self.model_i.vertices = self.model.vertices
            self.model_i.triangles = intersections
            self.updateShape("model_i")

            # === TASK 7 ===
            '''
            top, bottom = self.task_7_plane_split(self.model, self.plane)

            self.model_top.vertices = self.model.vertices + 0.2 * self.plane[:3]
            self.model_top.triangles = top
            self.model_top.vertex_normals = self.model.vertex_normals
            self.model_top.remove_unreferenced_vertices()
            self.updateShape("model_top")

            self.model_bottom.vertices = self.model.vertices
            self.model_bottom.triangles = bottom
            self.model_bottom.vertex_normals = self.model.vertex_normals
            self.model_bottom.remove_unreferenced_vertices()
            self.updateShape("model_bottom")
            '''

            # === TASK 8 ===
            precisely_cut_lines = self.task_8_exact_intersection(self.model.vertices[intersections], self.plane)
            self.removeShape('task-8_lines')
            self.addShape(precisely_cut_lines, 'task-8_lines')

            # === TASK 9 ===
            (   top_triangles,
                bottom_triangles,
                top_vertices,
                bottom_vertices
            ) = self.task_9_exact_split(self.model, self.plane)

            self.model_top.vertices = top_vertices + 0.005 * self.plane[:3]
            self.model_top.triangles = top_triangles
            self.model_top.remove_unreferenced_vertices()
            self.updateShape("model_top")

            self.model_bottom.vertices = bottom_vertices
            self.model_bottom.triangles = bottom_triangles
            self.model_bottom.remove_unreferenced_vertices()
            self.updateShape("model_bottom")

        return;

def intersect_triangle_with_plane(triangle: np.ndarray, n: np.ndarray, d: float) -> list[np.ndarray]:
    ''' n * x = d {Plane Vector Equation} '''

    distances = np.dot(triangle, n) - d # Όπως στο task_6_plane_intersection

    edges = [(0,1), (1,2), (2,0)] # Όλες οι ακμές του τριγώνου
    intersection_points = []
    for (i, j) in edges:
        d1 = distances[i]
        d2 = distances[j]
        if d1 * d2 < 0:
            t = d1 / (d1 - d2) # t = λ
            p = t * triangle[j] + (1 - t) * triangle[i]
            intersection_points.append(p)
    
    return intersection_points;

if __name__ == "__main__":
    app = Lab4()
    app.mainLoop()
