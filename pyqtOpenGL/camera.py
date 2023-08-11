from math import radians, tan
from enum import Enum, auto
from .transform3d import Matrix4x4, Quaternion, Vector3

class Camera:

    def __init__(
        self,
        position = Vector3(0., 0., 5),
        yaw = 0,
        pitch = 0,
        fov = 45,
    ):
        """View Corrdinate System
        default front vector: (0, 0, -1)
        default up Vector: (0, 1, 0)
        yaw: rotate around VCS y axis
        pitch: rotate around VCS x axis
        """
        self.pos = Vector3(position)
        self.quat = Quaternion.fromEulerAngles(pitch, yaw, 0.)
        self.fov = fov

    def get_view_matrix(self):
        return Matrix4x4.fromTranslation(-self.pos.x, -self.pos.y, -self.pos.z) * self.quat

    def get_projection_matrix(self, width, height, fov=None):
        distance = max(self.pos.z, 1)
        if fov is None:
            fov = self.fov

        return Matrix4x4.create_projection(
            fov,
            width / height,
            0.001 * distance,
            100.0 * distance
        )

    def get_proj_view_matrix(self, width, height, fov=None):
        return self.get_projection_matrix(width, height, fov) * self.get_view_matrix()

    def orbit(self, yaw, pitch):
        """Orbits the camera around the center position.
        *yaw* and *pitch* are given in degrees."""
        q =  Quaternion.fromEulerAngles(pitch, yaw, 0.)
        self.quat = q * self.quat

    def pan(self, dx, dy, dz=0.0, width=1000):
        """Pans the camera by the given amount of *dx*, *dy* and *dz*."""
        scale = self.pos.z * 2. * tan(0.5 * radians(self.fov)) / width
        self.pos += Vector3([-dx*scale, -dy*scale, dz*scale])
        if self.pos.z < 0.1:
            self.pos.z = 0.1

    def set_params(self, position=None, yaw=None, pitch=None, fov=None):
        if position is not None:
            self.pos = position
        if yaw is not None or pitch is not None:
            self.quat = Quaternion.fromEulerAngles(pitch, yaw, 0.)
        if fov is not None:
            self.fov = fov