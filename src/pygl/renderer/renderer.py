from ..scene.scene import Scene
from ..transforms import to_homogeneous
from .. import geometry
import numpy as np
import sdl2
import ctypes
import math

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

class Renderer:

    def __init__(self, scene: Scene, width: int = SCREEN_WIDTH, height: int = SCREEN_HEIGHT):
        self.scene = scene
        self.width = width
        self.height = height

        # initialize framebuffer with all-zeros (H, W, 3) RGB tensor
        self.framebuffer = np.zeros((height, width, 3), dtype=np.uint8)

        self.depthbuffer = np.full((self.height, self.width), np.inf)

    def rasterize(self):
        """
        Rasterize all `Mesh` within the scene by transforming the meshes vertices
        from the object space to the camera space according to `Scene`'s `Camera`
        placement, transforming into clipping space, doing perspective division,
        computing bbox, and doing pixel coverage test with depth testing. Each
        pixel RGB value is an interpolation of the triangle's vertices' RGB values,
        interpolated using barycentric weights of the pixel center.

        Every pixel RGB value is then stored in `Renderer.framebuffer`
        """

        for mesh in self.scene.meshes:

            vertices_object = to_homogeneous(mesh.vertices)

            vertices_world = vertices_object @ mesh.object2world_matrix().T

            view_matrix = self.scene.camera.world2object_matrix()

            vertices_camera = vertices_world @ view_matrix.T

            vertices_clip = vertices_camera @ self._projection_matrix().T

            # perspective division by w_clip = -z_c
            vertices_ndc = vertices_clip[:, :3] / vertices_clip[:, 3][:, None]

            vertices_screen = np.copy(vertices_ndc)

            # x inside-screen range: [-1, 1] => [0, self.width]
            # y inside-screen range: [-1, 1] => [0, self.height]
            vertices_screen[:, :2] = ((vertices_screen[:, :2] + 1) / 2) * np.array([[self.width, self.height]])

            faces = vertices_screen[mesh.faces]

            face_mins = faces.min(axis=1)
            face_maxes = faces.max(axis=1)

            # every face bbox, of shape (F, 4) with each entry being [x_min, x_max, y_min, y_max]
            face_bboxes = np.column_stack((np.clip(face_mins[:, 0], 0, self.width), 
                                           np.clip(face_maxes[:, 0], 0, self.width), 
                                           np.clip(face_mins[:, 1], 0, self.height), 
                                           np.clip(face_maxes[:, 1], 0, self.height)))

            for fidx, bbox in enumerate(face_bboxes):

                face = faces[fidx]

                # for each pixel in bbox (pair x, y), determine whether the center of pixel 
                # lies inside the triangle owning the bbox
                for x in range(math.ceil(bbox[0] - 0.5), math.floor(bbox[1] + 0.5)):
                    for y in range(math.ceil(bbox[2] - 0.5), math.floor(bbox[3] + 0.5)):

                        p = np.array([x + 0.5, y + 0.5])

                        e12p = geometry.orient2d(face[0], face[1], p)
                        e23p = geometry.orient2d(face[1], face[2], p)
                        e31p = geometry.orient2d(face[2], face[0], p)

                        # if the pixel belongs to the triangle
                        if e12p >= 0 and e23p >= 0 and e31p >= 0:
                            self.framebuffer[y, x] = np.full((3,), 255)
                            

    def _projection_matrix(self):

        aspect_ratio = self.width / self.height

        adj_to_ops = 1 / math.tan(math.radians(self.scene.camera.fovy_deg / 2))
        f, n = self.scene.camera.far, self.scene.camera.near

        return np.array([
            [adj_to_ops / aspect_ratio, 0, 0, 0],
            [0, adj_to_ops, 0, 0],
            [0, 0, (f + n) / (n - f), 2 * f * n / (n - f)],
            [0, 0, -1, 0],
        ])

    def render(self):

        self.rasterize()

        # reverse the y-direction so y=0 is at the top
        framebuffer = np.ascontiguousarray(self.framebuffer[::-1, :, :])

        height, width, _ = framebuffer.shape

        sdl2.SDL_Init(sdl2.SDL_INIT_VIDEO)

        window = sdl2.SDL_CreateWindow(
            b"Rasterizer",
            sdl2.SDL_WINDOWPOS_CENTERED,
            sdl2.SDL_WINDOWPOS_CENTERED,
            width,
            height,
            0,
        )

        renderer = sdl2.SDL_CreateRenderer(
            window,
            -1,
            sdl2.SDL_RENDERER_ACCELERATED,
        )

        texture = sdl2.SDL_CreateTexture(
            renderer,
            sdl2.SDL_PIXELFORMAT_RGB24,
            sdl2.SDL_TEXTUREACCESS_STATIC,
            width,
            height,
        )

        sdl2.SDL_UpdateTexture(
            texture,
            None,
            framebuffer.ctypes.data_as(ctypes.c_void_p),
            width * 3,
        )

        sdl2.SDL_RenderClear(renderer)
        sdl2.SDL_RenderCopy(renderer, texture, None, None)
        sdl2.SDL_RenderPresent(renderer)

        event = sdl2.SDL_Event()
        running = True

        while running:
            while sdl2.SDL_PollEvent(ctypes.byref(event)):
                if event.type == sdl2.SDL_QUIT:
                    running = False

        sdl2.SDL_DestroyTexture(texture)
        sdl2.SDL_DestroyRenderer(renderer)
        sdl2.SDL_DestroyWindow(window)
        sdl2.SDL_Quit()
