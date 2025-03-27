from .types import NDArray, List, Tuple, ColorType, Number
from .lineset2d import LineSet2D
from .pointset2d import PointSet2D

from numpy import ndarray
from pyglet.shapes import Circle
from pyglet.graphics import Group
from shapely import MultiPoint, concave_hull

from .line2d import Line2D   # New addition
from .point2d import Point2D # New addition

class Polygon2D(LineSet2D):
    '''A class used to represent an arbitrary polygon in 2D space.'''
    
    def __init__(self, points:None|PointSet2D|NDArray|List|Tuple=None, lines:None|NDArray|List|Tuple=None, width:Number=1, color:ColorType=(0, 0, 0), reorderIfNecessary:bool=False):
        '''Inits Polygon2D.

        Inits a Polygon2D containing `points` connected according to
        `lines`.
        
        If `lines` is `None`, the `points` will be connected in
        sequence i.e., (0, 1), (1, 2), ..., (n-1, n), (n, 0).
        
        If `points` is `None`, the polygon will be initialized empty.

        Args:
            points: The points of the polygon.
            lines: The indices in `points` that are connected by a
                line.
            width: The width of the displayed lines.
            color: The color of the displayed lines (RGB or RGBA).
            reorderIfNecessary: If `True` (and `lines` is `None`), the
                points will be reordered to attempt to make a
                non - self-intersecting polygon.
        '''

        if isinstance(points, PointSet2D):
            points = points.points

        if points is not None and lines is None:
            if isinstance(points, (ndarray, list, tuple)):
                if reorderIfNecessary:
                    mp = MultiPoint(tuple(map(tuple, points)))
                    ch = concave_hull(mp, ratio=0)
                    points = tuple(ch.exterior.coords)[:-1]
                line_amount = len(points)
                lines = [[i, i + 1] for i in range(len(points) - 1)]
                lines.append([len(points) - 1, 0])
        
        super().__init__(points, lines, width, color)

        self.convex_hull_edges    = [Line2D(p1, p2) for (p1, p2) in self.points[self.lines]] # New addition
        self.convex_hull_vertices = [Point2D(p) for (p, _) in self.points[self.lines]]       # New addition

        return;

    @staticmethod
    def create_from_lineset(lineset:LineSet2D, width:Number=1, color:ColorType=(0, 0, 0)) -> "Polygon2D":
        '''Creates a Polygon2D object from a Lineset2D object.

        Args:
            lineset: The lineset to be turned into a polygon.
            width: The width of the displayed lines.
            color: The color of the displayed lines (RGB or RGBA).

        Returns:
            The Polygon2D object created from the lineset.
        '''
        return Polygon2D(lineset._points, lineset._lines, width, color)
