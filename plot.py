import matplotlib.pyplot as plt
import numpy as np

def plot_vertices_3d(
    vertices_groups: np.ndarray,
    colors: np.ndarray | None = None,
    elev_deg: float = 15,
    azim_deg: float = 45
) -> None:

    """
    Plots multiple groups of 3D vertices.

    vertices_groups:
        Shape (G, N, 3), where G is the number of vertex groups.

    colors:
        Optional one-dimensional array of color strings with shape (G,).
        If provided, the number of colors must match the number of
        vertex groups.

    elev_deg:
        Camera elevation angle in degrees. Controls the vertical viewing
        angle of the 3D plot.

    azim_deg:
        Camera azimuth angle in degrees. Controls the horizontal rotation
        of the view around the vertical Y-axis.
    """

    if vertices_groups.ndim != 3 or vertices_groups.shape[2] != 3:
        raise ValueError(
            "'vertices_groups' must have shape (G, N, 3)"
        )

    num_groups = vertices_groups.shape[0]

    if colors is not None:
        if colors.ndim != 1:
            raise ValueError("'colors' must be a one-dimensional array")

        if len(colors) != num_groups:
            raise ValueError(
                "Number of colors must match number of vertex groups"
            )

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    for i, vertices in enumerate(vertices_groups):
        if colors is None:
            ax.scatter(
                vertices[:, 0],
                vertices[:, 1],
                vertices[:, 2],
            )
        else:
            ax.scatter(
                vertices[:, 0],
                vertices[:, 1],
                vertices[:, 2],
                color=colors[i],
            )

    # Symmetric range around the origin
    radius = max(np.abs(vertices_groups).max(), 1.0) * 1.1

    ax.set_xlim(-radius, radius)
    ax.set_ylim(-radius, radius)
    ax.set_zlim(-radius, radius)

    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()

    # Axes through the origin
    ax.plot([-radius, radius], [0, 0], [0, 0], color="black")
    ax.plot([0, 0], [-radius, radius], [0, 0], color="black")
    ax.plot([0, 0], [0, 0], [-radius, radius], color="black")

    # Integer ticks
    tick_min = int(np.ceil(-radius))
    tick_max = int(np.floor(radius))

    tick_size = radius * 0.025
    label_offset = radius * 0.05

    for t in range(tick_min, tick_max + 1):
        if t == 0:
            continue

        # X
        ax.plot(
            [t, t],
            [-tick_size, tick_size],
            [0, 0],
            color="black",
        )
        ax.text(t, -label_offset, 0, str(t))

        # Y
        ax.plot(
            [-tick_size, tick_size],
            [t, t],
            [0, 0],
            color="black",
        )
        ax.text(-label_offset, t, 0, str(t))

        # Z
        ax.plot(
            [0, 0],
            [-tick_size, tick_size],
            [t, t],
            color="black",
        )
        ax.text(0, -label_offset, t, str(t))

    # Origin
    ax.text(
        -label_offset,
        -label_offset,
        0,
        "0",
    )

    # Positive axis labels
    ax.text(radius, 0, 0, "+X")
    ax.text(0, radius, 0, "+Y")
    ax.text(0, 0, radius, "+Z")

    ax.view_init(
        elev=elev_deg,
        azim=azim_deg,
        vertical_axis="y",
    )

    plt.show()