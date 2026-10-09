from __future__ import annotations

from .camera import Camera
from .mesh import Mesh

from pathlib import Path
import numpy as np
import json


_BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
_SCENES_DIR = _BASE_DIR / "assets" / "scenes"

class Scene:

    def __init__(self, 
                 meshes: np.ndarray[tuple[int], Mesh], 
                 camera: Camera | None = None):
        
        """
        Receives scene's multiple meshes and camera

        If 'camera' is not provided, camera is set to default at pos (0, 0, 0) facing -Z
        """

        if camera is None:
            camera = Camera()
        
        self.set_camera(camera)
        self.meshes = meshes


    def set_camera(self, camera: Camera):
        self.camera = camera

    @staticmethod
    def load_scene(scene_name: str) -> Scene:

        """
        Loads a scene in assets/scenes/ based on the scene name and returns the
        corresponding Scene object.

        Scene file in /scenes/ is expected to be in JSON and must contain mandatory 'vertices_object'
        and 'faces' field, while the rest of the fields are optional (default to identity / no operation).
        """

        with open(_SCENES_DIR / (scene_name + ".json"), "r", encoding="utf-8") as file:

            data = json.load(file)

            assert 'vertices_object' in data and 'faces' in data

            vertices = np.array(data['vertices_object'], dtype=np.float64) 

            # expects ndarray of shape (N, 3)
            assert vertices.ndim == 2 and vertices.shape[1] == 3

            faces = np.array(data['faces'], dtype=np.int16)

            # expects ndarray of shape (F, 3)
            assert faces.ndim == 2 and faces.shape[1] == 3

            faces_attrs = data['faces_attributes']

            faces_attrs = {str(k): np.array(v) for k, v in faces_attrs.items()}

            translation = np.array(data['translation'] if 'translation' in data else [0, 0, 0])

            assert translation.ndim == 1 and translation.shape[0] == 3

            rotation_deg = np.array(data['rotation_deg'] if 'rotation_deg' in data else [0, 0, 0])

            assert rotation_deg.ndim == 1 and rotation_deg.shape[0] == 3

            rotation = np.deg2rad(rotation_deg)

            scale = np.array(data['scale'] if 'scale' in data else [1, 1, 1])

            assert scale.ndim == 1 and scale.shape[0] == 3

            mesh = Mesh(vertices, faces, faces_attrs, scale, rotation, translation)

            camera_center = np.array(data['camera_center'] if 'camera_center' in data else [0, 0, 0])
            
            assert camera_center.ndim == 1 and camera_center.shape[0] == 3

            camera_rotation_deg = np.array(data['camera_rotation_deg'] if 'camera_rotation_deg' in data else [0, 0, 0])

            assert camera_rotation_deg.ndim == 1 and camera_rotation_deg.shape[0] == 3

            camera_rotation = np.deg2rad(camera_rotation_deg)

            camera = Camera(camera_rotation, camera_center)

            return Scene(np.array([mesh]), camera)
