import numpy as np

from .. import transforms

class SceneObject:
    
    def __init__(self, 
                 scale: np.ndarray = np.array([1, 1, 1]),
                 rotation: np.ndarray = np.array([0, 0, 0]),
                 translation: np.ndarray = np.array([0, 0, 0])):

        assert scale.ndim == 1 and scale.shape[0] == 3
        assert rotation.ndim == 1 and rotation.shape[0] == 3
        assert translation.ndim == 1 and translation.shape[0] == 3

        self.scale = scale
        self.rotation = rotation
        self.translation = translation

    def scale_matrix(self):
        return np.diag(np.append(self.scale, 1))

    def rotation_matrix(self):
        return transforms.rotation_matrix(self.rotation[0], self.rotation[1], self.rotation[2])

    def translation_matrix(self):
        T = np.identity(4)
        T[:3, 3] = self.translation
        return T

    def object2world_matrix(self):
        return self.translation_matrix() @ self.rotation_matrix() @ self.scale_matrix()

    def world2object_matrix(self):
        return np.linalg.inv(self.object2world_matrix())
