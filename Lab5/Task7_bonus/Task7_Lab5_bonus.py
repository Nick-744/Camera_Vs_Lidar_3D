from vvrpywork.constants import Key, Color
from vvrpywork.scene import Scene3D, Scene3D_
from vvrpywork.shapes import Cuboid3D, PointSet3D, Mesh3D

import numpy as np

(WIDTH, HEIGHT) = (1000, 800)
COLORS = [Color.RED, Color.GREEN, Color.BLUE, Color.YELLOW, Color.ORANGE, Color.MAGENTA, Color.YELLOWGREEN, Color.CYAN]

''' Αν το ξανακοιτάξεις ποτέ...:

Η create_octree θα μπορούσε να δέχεται λίστα αντί για cube,
έτσι ώστε στην on_key_press να μην υπάρχει υπολογισμός από την αρχή!
Δηλαδή, το recursion να συνεχίζει από τα δεδομένα που έχει ήδη! '''

class Task7_Lab5(Scene3D_):
    def __init__(self):
        super().__init__(WIDTH, HEIGHT, "Task7-Lab5")
        self.reset()

        return;

    def reset(self):
        bunny_vertices = Mesh3D("resources/bunny_low.obj").vertices # Για να τελειώσει κιόλας κάποια στιγμή
                                                                    # το πρόγραμμα (είμαι στο potato pc...)!
        print(f"Vertices: {len(bunny_vertices)}")

        self.pcd = PointSet3D(bunny_vertices - np.mean(bunny_vertices, axis = 0))
        self.first_aabb = find_AABB(self.pcd.points)

        self.octree_ls = []
        self.octree_depth = -1

        self.addShape(self.first_aabb, 'aabb')
        self.addShape(self.pcd, 'pcd')

        return;

    def add_octree_cubes_in_scene(self) -> None:
        for (i, cube) in enumerate(self.octree_ls):
            self.addShape(cube, f'{i}')

        if self.octree_depth >= 0:
            self.removeShape('aabb')

        return;

    def remove_octree_cubes_from_scene(self) -> None:
        for i in range(len(self.octree_ls)):
            self.removeShape(f'{i}')

        if self.octree_depth == -1:
            self.addShape(self.first_aabb, 'aabb')

        return;

    def on_key_press(self, symbol, modifiers):
        if symbol == Key.UP:
            self.remove_octree_cubes_from_scene()
            self.octree_depth += 1
            
            self.octree_ls.clear()
            self.create_octree(
                self.first_aabb,
                self.pcd.points,
                0,
                self.octree_ls
            )
            self.add_octree_cubes_in_scene()

        if symbol == Key.DOWN:
            if self.octree_depth > -1:
                self.octree_depth -= 1
                self.remove_octree_cubes_from_scene()

                self.octree_ls.clear()
                self.create_octree(
                    self.first_aabb,
                    self.pcd.points,
                    0,
                    self.octree_ls
                )
                self.add_octree_cubes_in_scene()

        if (symbol == Key.UP) or (symbol == Key.DOWN):
            print(f"Octree depth: {self.octree_depth + 1}")

        return;

    def create_octree(self, cube: 'Cuboid3D', pcd: np.ndarray, depth: int, octree_ls: list[Cuboid3D]) -> None:
        if depth > self.octree_depth:
            return;

        children = cut_cube_in8cubes(cube)
        for child in children:
            if not points_in_cube_exist(child, pcd):
                continue; # Άδειος κύβος!
            
            if depth == self.octree_depth:
                child.color = COLORS[depth % len(COLORS)]
                octree_ls.append(child)
            
            self.create_octree(child, pcd, depth + 1, octree_ls)

        return;

def find_AABB(pcd: np.ndarray) -> Cuboid3D:
    vertices = pcd

    p1 = vertices.min(axis = 0)
    p2 = vertices.max(axis = 0)
    aabb = Cuboid3D(p1, p2) # Τα max και min, τα βρήκα προσημασμένα!
                            # Αυτά αποτελούν τις 2 άκρες του AABB!

    return aabb;

def cut_cube_in8cubes(cube: 'Cuboid3D') -> list['Cuboid3D']:
    xyz_min = [cube.x_min, cube.y_min, cube.z_min]
    xyz_max = [cube.x_max, cube.y_max, cube.z_max]
    temp_tuple = (xyz_min, xyz_max)

    corners = [np.array(xyz_min), np.array(xyz_max)]
    for (i, xyz) in enumerate(temp_tuple):
        for j in range(3):
            temp = xyz[:]
            temp[j] = temp_tuple[i - 1][j]
            corners.append(np.array(temp))

    center = (corners[0] + corners[1]) / 2
    # Μπορεί να χρειαστεί αργότερα η λογική...
    # edges_mid = []
    # for i in range(2, 5):
    #     edges_mid.append((corners[0] + corners[i]) / 2)
    # for i in range(5, 8):
    #     edges_mid.append((corners[1] + corners[i]) / 2)
    # for (i, j) in [(3, 5), (4, 6), (2, 7), (3, 7), (5, 4), (6, 2)]:
    #     edges_mid.append((corners[i] + corners[j]) / 2)

    return [Cuboid3D(center, corner, width = 2) for corner in corners];

def points_in_cube_exist(cube: 'Cuboid3D', pcd: np.ndarray) -> bool:
    temp = False
    for point in pcd:
        if (cube.x_min <= point[0] <= cube.x_max)      \
            and (cube.y_min <= point[1] <= cube.y_max) \
                and (cube.z_min <= point[2] <= cube.z_max):
            temp = True
            break;
    
    return temp;

if __name__ == "__main__":
    scene = Task7_Lab5()
    scene.mainLoop()