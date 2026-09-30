import numpy as np
import math

def to_homogeneous(vertices: np.ndarray) -> np.ndarray:

    """
    Receives n vertices in 3-dimensional euclidean space: 'vertices', of shape (n, 3) 

    Returns the same vertices in the homogeneous space, of shape (n, 4)
    """
        
    vertices_h = np.hstack([vertices, np.ones([vertices.shape[0], 1])])
    return vertices_h


def from_homogeneous(vertices_h: np.ndarray) -> np.ndarray:

    """
    Receives n vertices in homogeneous space, 'vertices_h', of shape (n, 4)

    Returns the vertices in 3-dimensional euclidean space, of shape (n, 3), with each vertex coordinate
    divided by w

    Hard reject if one of w is 0 by raising ValueError
    """

    w = vertices_h[:, 3]

    if np.any(w == 0):
        raise ValueError("'vertices_h' has 0 in the w component")

    # numpy ndarray broadcasting: w[:, None] of size Nx1 is broadcasted to Nx3, matching vertices_h[:, :3]
    return vertices_h[:, :3] / w[:, None]


def translation_matrix(t: np.ndarray) -> np.ndarray:

    """
    Forms a translation matrix given translation parameters 't': [tx, ty, tz]
    """

    t = t[:3]
    T = np.identity(4)
    T[:3, 3] = t
    return T


def rotation_matrix_x(alpha: float) -> np.ndarray:

    """
    Forms a rotation along x-axis matrix given rotation angle 'alpha' in radians
    """

    R = np.zeros([4, 4])
    R[0, 0] = R[3, 3] = 1
    R[1, 1] = R[2, 2] = math.cos(alpha)

    sin_alpha = math.sin(alpha)
    R[2, 1] = sin_alpha
    R[1, 2] = -sin_alpha

    return R


def rotation_matrix_y(alpha: float) -> np.ndarray:

    """
    Forms a rotation along y-axis matrix given rotation angle 'alpha' in radians
    """

    R = np.zeros([4, 4])
    R[1, 1] = R[3, 3] = 1
    R[0, 0] = R[2, 2] = math.cos(alpha)

    sin_alpha = math.sin(alpha)
    R[2, 0] = -sin_alpha
    R[0, 2] = sin_alpha

    return R


def rotation_matrix_z(alpha: float) -> np.ndarray:

    """
    Forms a rotation along z-axis matrix given rotation angle 'alpha' in radians
    """

    R = np.zeros([4, 4])
    R[2, 2] = R[3, 3] = 1

    R[0, 0] = R[1, 1] = math.cos(alpha)

    sin_alpha = math.sin(alpha)
    R[1, 0] = sin_alpha
    R[0, 1] = -sin_alpha

    return R


def rotation_matrix(alpha: float, beta: float, gamma: float) -> np.ndarray:

    """
    Forms a general rotation matrix that applies per-axis rotations by the following order:
    1. Apply rotation along x-axis by 'alpha'
    2. Apply rotation along y-axis by 'beta'
    3. Apply rotation along z-axis by 'gamma'
    """

    R = rotation_matrix_z(gamma) @ rotation_matrix_y(beta) @ rotation_matrix_x(alpha)
    return R


def scaling_matrix(s: np.ndarray) -> np.ndarray:

    """
    Forms a scaling matrix given scaling parameters 's': [sx, sy, sz]

    Hard reject if one of sx, sy, sz is zero by raising ValueError
    """

    s = s[:3]
    if np.any(s == 0):
        raise ValueError("Scaling parameters sx, sy, and sz in 's' cannot have value zero")

    return np.diag(np.append(s, 1))


def model_matrix(T: np.ndarray, R: np.ndarray, S: np.ndarray) -> np.ndarray:

    """
    Forms a model matrix given translation matrix 'T', rotation matrix 'R', and scaling matrix 'S'

    Transformations are done in this order: scaling -> rotation -> translation
    """

    return T @ R @ S


def object_to_world(vertices_object: np.ndarray, M: np.ndarray) -> np.ndarray:

    """
    Transforms vertices in object / local space to world space using 'M' model matrix

    Receives and returns vertices of shape (N, 4)
    """

    return (M @ vertices_object.T).T


def view_matrix(camera_center: np.ndarray, camera_rotation: np.ndarray) -> np.ndarray:

    """
    Forms a view matrix 'V' based on camera's center / position 
    'camera_center' and 'camera_rotation'
    """

    # inverse rotation matrix = transpose of orthonormal rotation matrix
    R_inv = rotation_matrix(*camera_rotation).T

    T_inv = translation_matrix(-camera_center) # inverse translation matrix

    # apply inverse translation first, then inverse rotation
    return R_inv @ T_inv


def world_to_camera(vertices_world: np.ndarray, V: np.ndarray) -> np.ndarray:

    """
    Transforms 'vertices_world' of size (N, 4) from world space to
    view / camera space
    """

    return (V @ vertices_world.T).T


def object_to_camera(vertices_object: np.ndarray, M: np.ndarray, V: np.ndarray) -> np.ndarray:

    """
    Convenient wrapper to transform vertices from object space to world space,
    then to camera space
    """

    return world_to_camera(object_to_world(vertices_object, M), V)
