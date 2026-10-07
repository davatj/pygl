from .scene_object import SceneObject
import numpy as np

class Camera(SceneObject):

    def __init__(self, 
                 rotation: np.ndarray = np.zeros(3), 
                 translation: np.ndarray = np.zeros(3),
                 fovy_deg: float = 120,
                 near: float = 1e-3,
                 far: float = 1e3):

        super().__init__(np.ones(3), rotation, translation)

        self.fovy_deg = fovy_deg
        self.near = near
        self.far = far
