from vvrpywork.constants import Key, Mouse, Color
from vvrpywork.scene import Scene3D, Scene3D_, get_rotation_matrix, world_space
from vvrpywork.shapes import (
    Point3D, Line3D, Arrow3D, Sphere3D, Cuboid3D, Cuboid3DGeneralized,
    PointSet3D, LineSet3D, Mesh3D, Label3D
)

from matplotlib import colormaps as cm
import numpy as np
import scipy
from scipy import sparse
import time

# from functools import cache

WIDTH = 1000
HEIGHT = 800

class Lab6(Scene3D_):
    def __init__(self):
        super().__init__(WIDTH, HEIGHT, "Lab6", output=True, n_sliders=2)
        self.reset_mesh()
        self.reset_sliders()
        self.printHelp()

        return;

    def reset_mesh(self):
        # Choose mesh
        obj_names = [
            "armadillo_low_low.obj",    # 0
            "bunny_low.obj",            # 1
            "cube.obj",                 # 2
            "dolphin.obj",              # 3
            "dragon_low_low.obj",       # 4
            "flashlight.obj",           # 5
            "flashlightNoCentered.obj", # 6
            "hand2.obj",                # 7
            "hand_clean.obj",           # 8
            "icosahedron.obj",          # 9
            "Phone_v02.obj",            # 10
            "pins.obj",                 # 11
            "polyhedron.obj",           # 12
            "suzanne.obj",              # 13
            "suzanne_clean.obj",        # 14
            "teapotMultiMesh.obj",      # 15
            "unicorn_low.obj",          # 16
            "unicorn_low_low.obj",      # 17
            "vvrlab.obj"                # 18
        ]
        #self.mesh = Mesh3D.create_bunny(color=Color.GRAY)
        mesh_name = obj_names[7]
        self.mesh = Mesh3D(f"resources/{mesh_name}", color=Color.RED)
        # self.mesh = Mesh3D("resources/dragon_low_low.obj", color=Color.GRAY)

        self.mesh.remove_duplicated_vertices()
        self.mesh.remove_unreferenced_vertices()
        vertices = self.mesh.vertices
        vertices -= np.mean(vertices, axis=0)
        distanceSq = (vertices ** 2).sum(axis=-1)
        max_dist = np.sqrt(np.max(distanceSq))
        self.mesh.vertices = vertices / max_dist
        self.removeShape("mesh")
        self.addShape(self.mesh, "mesh")

        self.wireframe = LineSet3D.create_from_mesh(self.mesh)
        self.removeShape("wireframe")
        self.addShape(self.wireframe, "wireframe")
        self.show_wireframe = True

        self.eigenvectors = None

        return;

    def reset_sliders(self):
        self.set_slider_value(0, 0)
        self.set_slider_value(1, 0.1)

        return;
        
    @world_space # Κάνει τα x, y, z να είναι οι συντεταγμένες του κόσμου και όχι του παραθύρου
    def on_mouse_press(self, x, y, z, button, modifiers):
        if button == Mouse.MOUSELEFT and modifiers & Key.MOD_SHIFT:
            if np.isinf(z):
                return; # Τότε δεν είμαστε πάνω στο mesh! Δουλεύει μέσω του Z buffer
            
            self.selected_vertex = find_closest_vertex(self.mesh, (x, y, z))

            vc = self.mesh.vertex_colors
            vc[self.selected_vertex] = (1, 0, 0)
            self.mesh.vertex_colors = vc # Γίνεται έτσι (και όχι απευθείας) για να δουλεύει σωστά!
            self.updateShape("mesh", True)

        return;

    def on_key_press(self, symbol, modifiers):
        if symbol == Key.R:
            self.reset_mesh()

        if symbol == Key.W:
            if self.show_wireframe:
                self.removeShape("wireframe")
                self.show_wireframe = False
            else:
                self.addShape(self.wireframe, "wireframe")
                self.show_wireframe = True
                
        if symbol == Key.A and hasattr(self, "selected_vertex"):
            adj = find_adjacent_vertices(self.mesh, self.selected_vertex)
            colors = self.mesh.vertex_colors
            for idx in adj:
                colors[idx] = (0, 0, 1)
            self.mesh.vertex_colors = colors
            self.updateShape("mesh")

        if symbol == Key.D and not modifiers & Key.MOD_CTRL and hasattr(self, "selected_vertex"):
            d = delta_coordinates_single(self.mesh, self.selected_vertex)
            self.print(d)

        if symbol == Key.D and modifiers & Key.MOD_CTRL:
            start = time.time()

            delta = []
            for i in range(len(self.mesh.vertices)):
                delta.append(delta_coordinates_single(self.mesh, i))

            self.print(f"Took {(time.time() - start):.3f} seconds.")

            self.display_delta_coords(np.array(delta)) # Για να τα χρωματίσουμε κιόλας!

        if symbol == Key.L:
            start = time.time()
            d_coords = delta_coordinates(self.mesh)
            self.print(f"Took {(time.time() - start):.3f} seconds.")

            self.display_delta_coords(d_coords)

        if symbol == Key.S:
            start = time.time()
            d_coords = delta_coordinates_sparse(self.mesh)
            self.print(f"Took {(time.time() - start):.3f} seconds.")

            self.display_delta_coords(d_coords)

        if symbol == Key.E:
            (vals, vecs) = eigendecomposition_full(self.mesh)
            self.eigenvectors = vecs
            self.display_eigenvector(vecs[:, self.eigenvector_idx])

            # --- Για την 4η άσκηση ---
            print('\nEigenvalues:')
            print(vals)
            print('\nEigenvectors:')
            print(vecs)
            print('\nΔιαστάσεις:')
            print(vecs.shape)
            print('\nΠλήθος κορυφών μοντέλου:')
            print(len(self.mesh.vertices))

        if symbol == Key.B:
            _, vecs = eigendecomposition_some(self.mesh, self.percent, "SM") # keep the smallest self.percent eigenvectors
            vertices = self.mesh.vertices
            new_vertices = vecs @ vecs.T @ vertices # reconstruct the mesh vertices
            self.mesh.vertices = new_vertices
            self.updateShape("mesh")

            self.wireframe.points = self.mesh.vertices
            self.updateShape("wireframe")

        if symbol == Key.C:
            _, vecs = eigendecomposition_some(self.mesh, self.percent, "LM")
            vertices = self.mesh.vertices
            new_vertices = vecs @ vecs.T @ vertices
            self.mesh.vertices = new_vertices
            self.updateShape("mesh")

            self.wireframe.points = self.mesh.vertices
            self.updateShape("wireframe")

        if symbol == Key.SLASH:
            self.printHelp()

        # --- Για το 5ο ερώτημα ---

        # a.
        if symbol == Key.N:
            iter = 3000
            lamb = 0.5

            self.mesh.vertices = laplacian_smoothing(self.mesh, iter, lamb)
            self.updateShape("mesh")

            self.wireframe.points = self.mesh.vertices
            # self.updateShape("wireframe")

        # b.
        if symbol == Key.Z:
            iter = 100
            lamb = +0.5
            mu   = -0.5

            self.mesh.vertices = taubin_smoothing(self.mesh, iter, lamb, mu)
            self.updateShape("mesh")

            self.wireframe.points = self.mesh.vertices
            # self.updateShape("wireframe")

        return;

    def on_slider_change(self, slider_id, value):
        if slider_id == 0:
            self.eigenvector_idx = int(value * (len(self.mesh.vertices) - 1))
            if self.eigenvectors is not None:
                self.display_eigenvector(self.eigenvectors[:, self.eigenvector_idx])

        if slider_id == 1:
            self.percent = value

        return;
    
    def printHelp(self):
        self.print("\
SHIFT+M1: Select vertex\n\
R: Reset mesh\n\
W: Toggle wireframe\n\
A: Adjacent vertices\n\
D: Delta coordinates single\n\
CTRL+D: Delta coordinates loop\n\
L: Delta coordinates laplacian\n\
S: Delta coordinates sparse\n\
E: Eigendecomposition\n\
B: Reconstruct from first {slider2}% eigenvetors\n\
C: Reconstruct from last {slider2}% eigenvetors\n\
?: Show this list\n\n")
        
        return;

    def display_delta_coords(self, delta: np.ndarray):
        norm = np.sqrt((delta * delta).sum(-1))

        # linear interpolation
        norm = (norm - norm.min()) / (norm.max() - norm.min()) if norm.max() - norm.min() != 0 else np.zeros_like(norm)
        
        colormap = cm.get_cmap("plasma")
        colors = colormap(norm)
        self.mesh.vertex_colors = colors[:,:3]
        self.updateShape("mesh")

        return;

    def display_eigenvector(self, vec: np.ndarray):
        # linear interpolation
        vec = (vec - vec.min()) / (vec.max() - vec.min()) if vec.max() - vec.min() != 0 else np.zeros_like(vec)
        
        colormap = cm.get_cmap("plasma")
        colors = colormap(vec)
        self.mesh.vertex_colors = colors[:,:3]
        self.updateShape("mesh")

        return;

def find_closest_vertex(mesh: Mesh3D, query: tuple) -> int:
    difference = (mesh.vertices - query)
    dist = (difference * difference).sum(axis = 1) # axis = 1, για να κάνουμε
                                                   # πράξεις σε κάθε γραμμή
    find_closest_vertex_index = np.argmin(dist)
    
    return find_closest_vertex_index;

def find_adjacent_vertices(mesh: Mesh3D, idx: int) -> np.ndarray:
    triangles = mesh.triangles

    adj = []
    adjacent_triangles = triangles[np.any(triangles == idx, axis = 1)]
    adj.extend(adjacent_triangles[adjacent_triangles != idx])
    adj = np.unique(adj)

    return adj; # Το np.unique κάνει το adj np.ndarray

# @cache
def delta_coordinates_single(mesh: Mesh3D, idx: int) -> np.ndarray:
    vertecies = mesh.vertices

    Ni = find_adjacent_vertices(mesh, idx)
    d_i = len(Ni)
    v_i = vertecies[idx]
    v_j = vertecies[Ni]

    return v_i - (1 / d_i) * v_j.sum(axis = 0);

def adjacency(mesh: Mesh3D) -> np.ndarray:
    triangles = mesh.triangles
    num_vertices = len(mesh.vertices)

    # Πίνακας γειτνίασης
    A = np.zeros((num_vertices, num_vertices), dtype = np.uint8)

    for t in triangles:
        (v1, v2, v3) = t
        A[v1, v2] = 1
        A[v2, v1] = 1
        A[v2, v3] = 1
        A[v3, v2] = 1
        A[v3, v1] = 1
        A[v1, v3] = 1

    return A; # unit8 ώστε να καταλαβάνει λιγότερη μνήμη!

def adjacency_sparse(mesh: Mesh3D) -> sparse.csr_array:
    num_vertices = len(mesh.vertices)
    triangles = mesh.triangles

    A = sparse.lil_array((num_vertices, num_vertices), dtype = np.uint8)

    for t in triangles:
        (v1, v2, v3) = t
        A[v1, v2] = 1
        A[v2, v1] = 1
        A[v2, v3] = 1
        A[v3, v2] = 1
        A[v3, v1] = 1
        A[v1, v3] = 1

    return A.tocsr();

def degree(A: np.ndarray) -> np.ndarray:
    D = np.zeros_like(A, dtype = np.uint8)
    np.fill_diagonal(D, A.sum(axis = 0)) # Δεν έχει σημασία το axis εδώ,
                                         # γιατί είναι συμμετρικός πίνακας!

    return D;

def degree_sparse(A: sparse.csr_array) -> sparse.csr_array:
    D = sparse.dia_array((A.sum(axis = 1), 0), shape = A.shape, dtype = np.uint8)

    return D.tocsr();

def diagonal_inverse(mat: np.ndarray) -> np.ndarray:
    d = np.diag(mat)
    
    return np.diag(1 / d);

def diagonal_inverse_sparse(mat: sparse.csr_array) -> sparse.csr_array:
    d = mat.diagonal()

    return sparse.dia_array((1 / d, 0), shape = mat.shape, dtype = np.float64).tocsr();

def random_walk_laplacian(mesh: Mesh3D) -> np.ndarray:
    A = adjacency(mesh)
    D = degree(A)
    D_inv = diagonal_inverse(D)
    I = np.eye(*A.shape) # Το * κάνει unpacking του tuple (num_vertices, num_vertices)!!!

    L_RW = I - D_inv @ A

    return L_RW;

def random_walk_laplacian_sparse(mesh: Mesh3D) -> sparse.csr_array:
    A = adjacency_sparse(mesh)
    D = degree_sparse(A)
    D_inv = diagonal_inverse_sparse(D)
    I = sparse.csr_array(sparse.eye(A.shape[0]))

    L_RW = I - D_inv @ A

    return L_RW;

def delta_coordinates(mesh: Mesh3D) -> np.ndarray:
    X = mesh.vertices
    L = random_walk_laplacian(mesh)

    return L @ X;

def delta_coordinates_sparse(mesh: Mesh3D) -> np.ndarray:
    X = mesh.vertices
    L = random_walk_laplacian_sparse(mesh)

    return L @ X;

def graph_laplacian_sparse(mesh: Mesh3D) -> sparse.csr_array:
    A = adjacency_sparse(mesh).astype(np.int8)
    D = degree_sparse(A)

    return D - A;

def eigendecomposition_full(mesh: Mesh3D) -> tuple[np.ndarray, np.ndarray]:
    L = graph_laplacian_sparse(mesh)

    return scipy.linalg.eigh(L._asfptype().toarray());

def eigendecomposition_some(mesh: Mesh3D, keep_percentage=0.1, which="SM") -> tuple[np.ndarray, np.ndarray]:
    L = graph_laplacian_sparse(mesh)

    k = int(L.shape[0] * keep_percentage)
    k = 1 if k == 0 else k # Για να μην έχουμε 0 eigenvectors

    return sparse.linalg.eigsh(L._asfptype().toarray(), k = k, which = which)

# --- Ερώτημα 5ο | Lab 6 ---

# a.
def laplacian_smoothing(mesh: Mesh3D, iterations: int = 10, lambda_const: float = 0.1) -> np.ndarray:
    L = random_walk_laplacian_sparse(mesh)
    I = sparse.eye(L.shape[0])
    temp = mesh.vertices.copy()

    for _ in range(iterations):
        temp = (I - lambda_const * L) @ temp

    return temp;

# b.
def taubin_smoothing(mesh: Mesh3D, iterations: int = 10,
                     lambda_const: float = 0.1, mu_const: float = -0.1) -> np.ndarray:
    L = random_walk_laplacian_sparse(mesh)
    I = sparse.eye(L.shape[0])
    temp = mesh.vertices.copy()

    for _ in range(iterations):
        temp = (I + lambda_const * L) @ temp # Shrink
        temp = (I + mu_const     * L) @ temp # Inflate

    return temp;

if __name__ == "__main__":
    app = Lab6()
    app.mainLoop()
