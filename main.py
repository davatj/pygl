from scene import load_scene
from plot import plot_vertices_3d
import transform
import numpy as np

def main():
    scene = load_scene("camera_offset")

    T = transform.translation_matrix(scene.translation)
    R = transform.rotation_matrix(scene.rotation[0], scene.rotation[1], scene.rotation[2])
    S = transform.scaling_matrix(scene.scale)
    M = transform.model_matrix(T, R, S)

    camera_rotation_rads = np.deg2rad(scene.camera_rotation_degs)
    R_cam = transform.rotation_matrix(*camera_rotation_rads)

    V = transform.view_matrix(scene.camera_center, R_cam)

    vertices_object_homogeneous = transform.to_homogeneous(scene.vertices_object)

    vertices_view_homogeneous = transform.object_to_camera(vertices_object_homogeneous, M, V)

    vertices_view = transform.from_homogeneous(vertices_view_homogeneous)

    # group two collection of vertices of size (N, 3) to (2, N, 3)
    vertices = np.stack((scene.vertices_object, vertices_view))

    plot_vertices_3d(vertices, np.array(["red", "blue"]), 0, 0)


if __name__ == "__main__":
    main()