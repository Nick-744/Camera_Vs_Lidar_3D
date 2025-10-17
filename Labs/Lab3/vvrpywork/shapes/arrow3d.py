from .types import NDArray3, List3, Tuple3, ColorType, Number
from .line3d import Line3D
from .point3d import Point3D
from vvrpywork.scene import Scene3D

from math import acos
import numpy as np
import open3d as o3d
import open3d.visualization.rendering as rendering


class Arrow3D(Line3D):
    '''A class used to represent an arrow in 3D space.'''

    def __init__(self, start:Point3D|NDArray3|List3|Tuple3, end:Point3D|NDArray3|List3|Tuple3, width:Number=1, resolution:int=20, color:ColorType=(0, 0, 0), cone_to_cylinder_ratio:Number=0.1):
        '''Inits Arrow3D given the arrow's start and end.

        Args:
            start: The coordinates of the arrow's start.
            end: The coordinates of the arrow's end.
            width: The width of the displayed arrow.
            resolution: The resolution of the displayed arrow.
            color: The color of the displayed arrow (RGB or RGBA).
            cone_to_cylinder_ratio: the percentage of the arrow's
                length that is taken up by the arrow head.
        '''
        super().__init__(start, end, width, resolution, color)
        self._ratio = cone_to_cylinder_ratio

    def _addToScene(self, scene:Scene3D, name:None|str):
        name = str(id(self)) if name is None else name
        scene._shapeDict[name] = self
        shape = o3d.geometry.TriangleMesh.create_arrow(0.005 * self.width, 2 * 0.005 * self.width, (1 - self.cone_to_cylinder_ratio) * self.length(), self.cone_to_cylinder_ratio * self.length(), self.resolution)
        shape.translate((0, 0, -self.length()/2))
        shape.compute_vertex_normals()

        v = np.array(((self.x2-self.x1)/self.length(), (self.y2-self.y1)/self.length(), (self.z2-self.z1)/self.length()))
        z = np.array((0., 0., 1.))
        a = np.cross(z, v)
        div = (a**2).sum()**0.5
        
        if div != 0:
            a /= div
            aa = a * acos(np.dot(z, v))

            self._rot = o3d.geometry.get_rotation_matrix_from_axis_angle(aa)
            rotation = np.zeros((4, 4))
            rotation[:3, :3] = self._rot
            rotation[3, 3] = 1
        else:
            self._rot = None
            rotation = np.array(((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)))

        translation = np.array(((1, 0, 0, (self.x1+self.x2)/2), (0, 1, 0, (self.y1+self.y2)/2), (0, 0, 1, (self.z1+self.z2)/2), (0, 0, 0, 1)))

        material = rendering.MaterialRecord()
        material.shader = "defaultLitTransparency"
        color = self.color
        color = tuple((*color, 1)) if len(color) == 3 else color
        material.base_color = color
        scene._scene_widget.scene.add_geometry(name, shape, material)
        scene._scene_widget.scene.set_geometry_transform(name, translation @ rotation)

    @property
    def cone_to_cylinder_ratio(self) -> Number:
        '''The percentage of the arrow's length that is taken up by the arrow head.'''
        return self._ratio
