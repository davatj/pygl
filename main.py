from scene import load_scene
from plot import plot_vertices_3d
import transforms
import numpy as np

def main():
    scene = load_scene("camera_offset")

    T = transforms.translation_matrix(scene.translation)

    rotation_rad = np.deg2rad(scene.rotation_deg)

    R = transforms.rotation_matrix(*rotation_rad)
    S = transforms.scaling_matrix(scene.scale)
    M = transforms.model_matrix(T, R, S)

    camera_rotation_rad = np.deg2rad(scene.camera_rotation_deg)

    V = transforms.view_matrix(scene.camera_center, camera_rotation_rad)

    vertices_object_homogeneous = transforms.to_homogeneous(scene.vertices_object)

    vertices_view_homogeneous = transforms.object_to_camera(vertices_object_homogeneous, M, V)

    vertices_view = transforms.from_homogeneous(vertices_view_homogeneous)

    # group two collection of vertices of size (N, 3) to (2, N, 3)
    vertices = np.stack((scene.vertices_object, vertices_view))

    plot_vertices_3d(vertices, np.array(["red", "blue"]))


if __name__ == "__main__":
    main()