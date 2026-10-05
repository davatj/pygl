from .scene_object import SceneObject
import numpy as np

class Camera(SceneObject):

    def __init__(self, 
                 rotation: np.ndarray = np.zeros(3), 
                 translation: np.ndarray = np.zeros(3)):

        super().__init__(np.ones(3), rotation, translation)
