from .scene_object import SceneObject
import numpy as np

class Mesh(SceneObject):

    def __init__(self, vertices: np.ndarray, faces: np.ndarray,
                 scale: np.ndarray = np.ones(3),
                 rotation: np.ndarray = np.zeros(3),
                 translation: np.ndarray = np.zeros(3)):

        super().__init__(scale, rotation, translation)

        # asserts that vertices has size (V, 3)
        assert vertices.ndim == 2 and vertices.shape[1] == 3

        # asserts that faces has size (F, 3)
        assert faces.ndim == 2 and faces.shape[1] == 3

        self.vertices = vertices
        self.faces = faces
