from pathlib import Path
import numpy as np
import json

BASE_DIR = Path(__file__).resolve().parent
SCENE_DIR = BASE_DIR / "scenes"

class Scene:
    def __init__(self, 
                 vertices_object: np.ndarray, 
                 faces: np.ndarray, 
                 translation: np.ndarray, 
                 rotation_deg: np.ndarray, 
                 scale: np.ndarray, 
                 camera_center: np.ndarray, 
                 camera_rotation_deg: np.ndarray):
        
        self.vertices_object = vertices_object
        self.faces = faces
        self.translation = translation
        self.rotation_deg = rotation_deg
        self.scale = scale
        self.camera_center = camera_center
        self.camera_rotation_deg = camera_rotation_deg


def load_scene(scene_name: str) -> Scene:

    """
    Loads scene in /scenes/ based on the scene name and returns Scene object representing the scene's
    data, transformations, and camera configuration.

    Scene file in /scenes/ is expected to be in JSON and must contain mandatory 'vertices_object'
    and 'faces' field, while the rest of the fields are optional (default to identity / no operation).
    """

    with open(SCENE_DIR / (scene_name + ".json"), "r", encoding="utf-8") as file:

        data = json.load(file)

        assert 'vertices_object' in data and 'faces' in data

        vertices_object = np.array(data['vertices_object'], dtype=np.float64) 

        # expects ndarray of shape (N, 3)
        assert vertices_object.ndim == 2 and vertices_object.shape[1] == 3

        faces = np.array(data['faces'], dtype=np.int16)

        # expects ndarray of shape (F, 3)
        assert faces.ndim == 2 and faces.shape[1] == 3

        translation = np.array(data['translation'] if 'translation' in data else [0, 0, 0])

        assert translation.shape[0] == 3

        rotation_deg = np.array(data['rotation_deg'] if 'rotation_deg' in data else [0, 0, 0])

        assert rotation_deg.shape[0] == 3

        scale = np.array(data['scale'] if 'scale' in data else [1, 1, 1])

        assert scale.shape[0] == 3

        camera_center = np.array(data['camera_center'] if 'camera_center' in data else [0, 0, 0])
        
        assert camera_center.shape[0] == 3

        camera_rotation_deg = np.array(data['camera_rotation_deg'] if 'camera_rotation_deg' in data else [0, 0, 0])

        assert camera_rotation_deg.shape[0] == 3

        return Scene(vertices_object, faces, translation, rotation_deg, scale, camera_center, camera_rotation_deg)