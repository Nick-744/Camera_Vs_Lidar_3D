from vvrpywork.constants import Key, Mouse, Color
from vvrpywork.scene import Scene2D
from vvrpywork.shapes import (
    Point2D, Line2D, Triangle2D, Circle2D, Rectangle2D,
    PointSet2D, LineSet2D, Polygon2D
)

from copy import deepcopy
from random import random
import numpy as np

WIDTH = 800
HEIGHT = 800

class Lab3(Scene2D):
    def __init__(self):
        super().__init__(WIDTH, HEIGHT, "Lab3")
        self.points:list[Point2D] = [] 
        self.triangles:dict[str, Triangle2D] = {}

        self.reset()

        return;

    def reset(self):
        self.violations = 0 # Πόσες φορές έχουμε βρει παραβίαση Delaunay
        self.run_task_3 = False
        self.run_task_4 = False
        self.run_task_5 = False
        self.run_task_6 = False
        self.task_6_add_on = False

        A = Point2D((-0.8, -0.8))
        B = Point2D((0.8, -0.8))
        C = Point2D((0, 0.8))

        self.points.append(A)
        self.points.append(B)
        self.points.append(C)

        big_triangle = Triangle2D(A, B, C)
        name = str(random())
        self.triangles[name] = big_triangle

        self.addShape(A)
        self.addShape(B)
        self.addShape(C)

        self.addShape(big_triangle, name)

        return;

    def on_mouse_press(self, x, y, button, modifiers):
        # Set all previous triangles to black
        for name in self.triangles:
            self.triangles[name].color = Color.BLACK
            self.updateShape(name)

        # Process new point
        self.processPoint(Point2D((x, y)))

        return;

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        new_pt = Point2D((x, y))
        if self.points[-1].distanceSq(new_pt) > 0.05:
            self.processPoint(new_pt)

        return;

    def on_key_press(self, symbol, modifiers):
        if symbol == Key._3:
            self.run_task_3 = True
            self.processPoint()
        if symbol == Key._4:
            self.run_task_4 = True
            self.processPoint()
        if symbol == Key._5:
            self.run_task_5 = True
            self.processPoint()
        if symbol == Key._6:
            self.run_task_6 = True
            self.processPoint()
        if symbol == Key._7:
            self.task_6_add_on = True
            self.processPoint()

        return;

    def clearViolationShapes(self):
        for i in range(self.violations):
            self.removeShape(f"vc{i}")
            self.removeShape(f"vt{i}")
        self.violations = 0

        return;

    def processPoint(self, point:Point2D|None = None):
        if not self.run_task_3 and not self.run_task_4 and not self.run_task_5 \
            and not self.run_task_6 and not self.task_6_add_on:
            # [Check 1]
            # Check whether a point already exists in the same coords.
            for p in self.points:
                if p.x == point.x and p.y == point.y:
                    return;
                
            # Find enclosing triangle.
            count_enclosing = 0
            for name in self.triangles:
                if self.triangles[name].contains(point):
                    count_enclosing += 1
                    name_enclosing = name

            # [Check 2]
            # If no enclosing triangle was found.
            # Or if more than one were found.
            if count_enclosing != 1:
                return;

            self.points.append(point)
            self.addShape(point)

            # Remove shapes from last click
            self.clearViolationShapes()


            # TASK 0:
            #   - Create the 3 subdivision triangles and store them to `new_triangles`.
            #   - Remove the enclosing triangle and store the new ones with different colour.
            #
            # HINTS:
            #   - To delete a triangle from `self.triangles`:
            #       del self.triangles[key_to_delete]
            #
            # VARIABLES:
            #   - point: Point2D                   # New point
            #   - name_enclosing: str              # key of the enclosing triangle
            #                                        (both in self.triangles and in the scene)
            #   - new_triangles: list[Triangle2D]  # The 3 triangles that will replace the enclosing.

            self.new_triangles:list[Triangle2D] = []

            # example
            #new_triangles.append(Triangle2D(point, (point.x + 0.1, point.y), (point.x, point.y + 0.1), color=Color.RED))
            #name = str(random())
            #self.triangles[name] = new_triangles[0]
            #self.addShape(new_triangles[0], name)

            # 1. Parse the 3 vertices of the enclosing triangle
            v1 = self.triangles[name_enclosing].getPoint1()
            v2 = self.triangles[name_enclosing].getPoint2()
            v3 = self.triangles[name_enclosing].getPoint3()
            # 2. Create the 3 new triangles and store them in the "new_triangles" list
            temp = [(v1, v2), (v2, v3), (v3, v1)]
            for (p1, p2) in temp:
                self.new_triangles.append(Triangle2D(point, p1, p2, color = Color.RED))
            # 3. Remove the enclosing triangle
            self.removeShape(name_enclosing)
            del self.triangles[name_enclosing]
            # 4. Add the new triangles in dict and scene
            for triangle in self.new_triangles:
                name = str(random())
                self.triangles[name] = triangle
                self.addShape(triangle, name)

            # TASK 2:
            #   - Check the 3 new triangles for Delaunay violations.
            #   - If not Delaunay, add it with different color and show CircumCircle.
            #
            # HINTS:
            #   - use isDelaunay()
                
            for new_triangle in self.new_triangles:
                if findViolations(self.triangles, new_triangle):
                    showViolations(new_triangle, self, (1, 1, 1, 0.5))

# ------------- Υλοποίηση των tasks 3, 4, 5 -------------

        try:
            if self.run_task_3:
                self.flip_violating_triangles(self.new_triangles)
                self.new_triangles.clear()

                self.run_task_3 = False

            if self.run_task_4:
                while True:
                    violating_triangles = [
                        triangle for (name, triangle) in self.triangles.items()
                        if findViolations(self.triangles, triangle)
                    ]

                    if not violating_triangles:
                        break;

                    self.flip_violating_triangles(violating_triangles)
                
                self.new_triangles.clear()
                self.clearViolationShapes()
                
                self.run_task_4 = False
        except:
            self.run_task_3 = False
            self.run_task_4 = False
            print("Δεν έχεις προσθέσει ακόμα σημεία!")

        voronoi_scene_names = ["voronoi_lines", "voronoi_points"]
        for name in voronoi_scene_names:
            self.removeShape(name)
        
        if self.run_task_5:
            voronoi_lines, voronoi_points = create_voronoi_diagram(self.triangles)
            voronoi_idx_lines = convert_lines_to_indices(voronoi_points, voronoi_lines)
            
            voronoi_lineSet = LineSet2D(points = voronoi_points, lines = voronoi_idx_lines, color = Color.RED)
            voronoi_pointSet = PointSet2D(voronoi_points, size = 1.5, color = Color.GREEN)
            for (index, voronoi_set) in enumerate([voronoi_lineSet, voronoi_pointSet]):
                self.addShape(voronoi_set, voronoi_scene_names[index])

            self.clearViolationShapes()
            
            self.run_task_5 = False

# ------------- Bonus | Task 6 -------------

        if self.run_task_6: # Key._6
            for _ in range(10):
                (biggest_t_key, biggest_t) = max(
                    self.triangles.items(), key = lambda item: item[1].get_area()
                ) # Βρες το τρίγωνο με το μεγαλύτερο εμβαδόν!

                self.subdivide_triangle_into_4(biggest_t_key, biggest_t)            
                
            self.clearViolationShapes()

            self.run_task_6 = False
        
        if self.task_6_add_on: # Key._7
            '''
            Για την απαίτηση:

            "Πραγματοποιήστε κατάλληλη επεξεργασία στα γειτονικά 
            τρίγωνα ώστε κάθε πλευρά των τριγώνων να έχει μόνο
            ένα γειτονικό τρίγωνο."
            '''
            while True:
                temp_all_triangles = deepcopy(self.triangles)
                for key, triangle in temp_all_triangles.items():
                    if not get_triangle_adjacency_count(self.triangles, triangle):
                        self.subdivide_triangle_into_4(key, triangle)
                        break;
                else:
                    break ;

            self.task_6_add_on = False

        return;

    def subdivide_triangle_into_4(self, triangle_key: str, triangle: Triangle2D):
        (m1, m2, m3) = get_medial_triangle_vertices(triangle)
        temp = [(m3, m1), (m1, m2), (m2, m3)]

        self.new_triangles:list[Triangle2D] = []
        for (i, vertex) in enumerate((triangle.getPoint1(), triangle.getPoint2(), triangle.getPoint3())):
            (p1, p2) = temp[i]
            self.new_triangles.append(Triangle2D(vertex, p1, p2, color = Color.RED))

        self.new_triangles.append(Triangle2D(m1, m2, m3, color = Color.RED))

        self.removeShape(triangle_key)
        del self.triangles[triangle_key]

        for triangle in self.new_triangles:
            name = str(random())
            self.triangles[name] = triangle
            self.addShape(triangle, name)

        return;

    def getTriangleNameByVertices(self, p1:Point2D, p2:Point2D, p3:Point2D) -> str:
        for key in self.triangles:
            t = self.triangles[key]
            # Η μέθοδος contains(), ελέγχει αν το σημείο είναι
            # μέσα ή πάνω στο περίγραμμα του τριγώνου!
            if t.contains(p1) and t.contains(p2) and t.contains(p3):
                return key;

        return "";

    def flip_violating_triangles(self, triangles_to_check: list[Triangle2D]):
        for triangle in triangles_to_check:
            if not findViolations(self.triangles, triangle):
                continue;
            
            (v1, v2, v3) = (triangle.getPoint1(), triangle.getPoint2(), triangle.getPoint3())
            triangle_name = self.getTriangleNameByVertices(v1, v2, v3)
            temp = [(v1, v2), (v1, v3), (v2, v3)]
            remaining_vertex = [v3, v2, v1]

            for (i, (p1, p2)) in enumerate(temp):
                tri_adj_key, opp_ver = findAdjacentTriangle(self.triangles, p1, p2)
                if not isDelaunay(triangle, opp_ver):
                    for key in [triangle_name, tri_adj_key]:
                        self.removeShape(key)
                        del self.triangles[key]

                    t1 = Triangle2D(p1, opp_ver, remaining_vertex[i], color = Color.BLUE)
                    t2 = Triangle2D(p2, opp_ver, remaining_vertex[i], color = Color.BLUE)
                    for t in [t1, t2]:
                        name = str(random())
                        self.triangles[name] = t
                        self.addShape(t, name)

                    break;

        return;

# ----------- Συναρτήσεις -----------

def get_triangle_adjacency_count(all_triangles: dict[str, Triangle2D], triangle: Triangle2D) -> bool:
    v1, v2, v3 = triangle.getPoint1(), triangle.getPoint2(), triangle.getPoint3()
    m1, m2, m3 = get_medial_triangle_vertices(triangle)
    triangle_points = {
        (v1.x, v1.y), (v2.x, v2.y), (v3.x, v3.y),
        (m1[0], m1[1]), (m2[0], m2[1]), (m3[0], m3[1])
    }
    triangle_vertices = {(v1.x, v1.y), (v2.x, v2.y), (v3.x, v3.y)}

    adjacency_count = 0
    adjacency_vertex_based = 0
    for other in all_triangles.values():
        if other is triangle:
            continue;

        ov1, ov2, ov3 = other.getPoint1(), other.getPoint2(), other.getPoint3()
        om1, om2, om3 = get_medial_triangle_vertices(other)
        other_points = {
            (ov1.x, ov1.y), (ov2.x, ov2.y), (ov3.x, ov3.y),
            (om1[0], om1[1]), (om2[0], om2[1]), (om3[0], om3[1])
        }
        other_vertices = {(ov1.x, ov1.y), (ov2.x, ov2.y), (ov3.x, ov3.y)}

        # https://www.w3schools.com/python/ref_set_intersection.asp
        shared = triangle_points.intersection(other_points)
        if len(shared) >= 2:
            adjacency_count += 1

        shared_vertices = triangle_vertices.intersection(other_vertices)
        if len(shared_vertices) >= 2:
            adjacency_vertex_based += 1

    return adjacency_vertex_based == adjacency_count;

def create_voronoi_diagram(triangles:dict[str,Triangle2D]) -> tuple[list, list]:
    no_duplicates_lines = list()
    no_duplicates_points = list()
    
    for (i, triangle) in enumerate(triangles.values()):
        barycenter1 = get_barycenter_tuple_wrapper(triangle)
        if barycenter1 not in no_duplicates_points:
            no_duplicates_points.append(barycenter1)
        
        (v1, v2, v3) = (triangle.getPoint1(), triangle.getPoint2(), triangle.getPoint3())

        temp = [(v1, v2), (v1, v3), (v2, v3)]
        for (p1, p2) in temp:
            tri_adj_key, _ = findAdjacentTriangle(triangles, p1, p2)
            barycenter2 = get_barycenter_tuple_wrapper(triangles[tri_adj_key])
            if barycenter2 == barycenter1:
                continue;
            
            test = (barycenter1, barycenter2)
            if (test not in no_duplicates_lines) and (test[::-1] not in no_duplicates_lines):
                no_duplicates_lines.append((barycenter1, barycenter2))

    return no_duplicates_lines, no_duplicates_points;

def convert_lines_to_indices(point_list, line_pairs) -> list[tuple[int, int]]:
    '''Μετατρέπει γραμμές (ως ζεύγοι σημείων) σε γραμμές με δείκτες του point_list, για χρήση στο LineSet2D.'''
    
    point_to_index = {point: i for (i, point) in enumerate(point_list)}
    indexed_line_pairs = [(point_to_index[p1], point_to_index[p2]) for (p1, p2) in line_pairs]

    return indexed_line_pairs;

def get_medial_triangle_vertices(t:Triangle2D) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # https://el.wikipedia.org/wiki/%CE%A3%CF%85%CE%BC%CF%80%CE%BB%CE%B7%CF%81%CF%89%CE%BC%CE%B1%CF%84%CE%B9%CE%BA%CF%8C_%CF%84%CF%81%CE%AF%CE%B3%CF%89%CE%BD%CE%BF

    v1 = t.getPoint1()
    v2 = t.getPoint2()
    v3 = t.getPoint3()
    v1 = np.array([v1.x, v1.y])
    v2 = np.array([v2.x, v2.y])
    v3 = np.array([v3.x, v3.y])

    return (v1 + v2) / 2, (v2 + v3) / 2, (v3 + v1) / 2;

def get_barycenter_tuple_wrapper(t:Triangle2D) -> tuple[float, float]:
    return tuple(get_barycenter(t));

def get_barycenter(t:Triangle2D) -> np.ndarray:
    '''Υπολόγισε το βαρυκέντρο του τριγώνου με σημείο αναφοράς την αρχή των αξόνων!!!'''
    # https://el.wikipedia.org/wiki/%CE%92%CE%B1%CF%81%CF%8D%CE%BA%CE%B5%CE%BD%CF%84%CF%81%CE%BF_%CF%84%CF%81%CE%B9%CE%B3%CF%8E%CE%BD%CE%BF%CF%85

    x_sum = t.getPoint1().x + t.getPoint2().x + t.getPoint3().x
    y_sum = t.getPoint1().y + t.getPoint2().y + t.getPoint3().y
    OG = (1 / 3) * np.array([x_sum, y_sum])

    return OG;

def findAdjacentTriangle(tris:dict[str,Triangle2D], p1:Point2D, p2:Point2D) -> tuple[str,Point2D]:
    tri_adj_key = ""
    opp_ver = None

    # TASK 1:
    #   - Find a triangle that contains p1-p2.
    #   - Save its key in `tri_adj_key` and the remaining vertex in `opp_ver`.

    p1 = (p1.x, p1.y)
    p2 = (p2.x, p2.y)
    for key in tris:
        v1 = tris[key].getPoint1()
        v2 = tris[key].getPoint2()
        v3 = tris[key].getPoint3()
        v1 = (v1.x, v1.y)
        v2 = (v2.x, v2.y)
        v3 = (v3.x, v3.y)

        temp = [v1, v2, v3]
        if (p1 in temp) and (p2 in temp):
            tri_adj_key = key
            opp_ver = [Point2D(p3) for p3 in temp if p3 != p1 and p3 != p2][0]

            break;

    return tri_adj_key, opp_ver;

def isDelaunay(t:Triangle2D, p:Point2D) -> bool:
    '''Checks if `t` is a Delaunay triangle w.r.t `p`.'''

    c = t.getCircumCircle()
    c.radius *= 0.99  # Shrink the circle a bit in order to exclude points of its circumference.
    if c.contains(p):
        return False;

    return True;

def findViolations(all_triangles:dict[str,Triangle2D], new_triangle:Triangle2D) -> bool:
    '''Checks if the given triangle is Delaunay.

    Checks if a triangle is delaunay, checking all its adjacent
    triangles.

    Args:
        all_triangles: A dictionary of all the triangles.
        new_triangle: The triangle to check.
    
    Returns:
        False if the given triangle is delaunay and True otherwise.
    '''

    is_delaunay = True

    # 1. Use findAdjacentTriangle() to check whether new_triangle is Delauney
    # 2. Use function isDelaunay()
        
    v1 = new_triangle.getPoint1()
    v2 = new_triangle.getPoint2()
    v3 = new_triangle.getPoint3()
    temp = [(v1, v2), (v1, v3), (v2, v3)]

    check_points = []
    for p1, p2 in temp:
        check_points.append(findAdjacentTriangle(all_triangles, p1, p2))

    for (name, p3) in check_points:
        if not isDelaunay(new_triangle, p3):
            is_delaunay = False

    return not is_delaunay;

def showViolations(tri:Triangle2D, scene:Lab3, col:Color):
    c = tri.getCircumCircle()
    c.color = col
    scene.addShape(c, f"vc{scene.violations}")

    filled = Triangle2D(tri.getPoint1(), tri.getPoint2(), tri.getPoint3(), color=col, filled=True)
    scene.addShape(filled, f"vt{scene.violations}")
    scene.violations += 1

    return;

if __name__ == "__main__":
    app = Lab3()
    app.mainLoop()
