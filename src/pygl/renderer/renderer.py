from ..scene.scene import Scene
import numpy as np
import sdl2
import ctypes

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

class Renderer:

    def __init__(self, scene: Scene, width: int = SCREEN_WIDTH, height: int = SCREEN_HEIGHT):
        self.scene = scene
        self.width = width
        self.height = height

        # initialize framebuffer with all-zeros (H, W, 3) RGB tensor
        self.framebuffer = np.zeros((height, width, 3), dtype=np.uint8)

        self.framebuffer = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

    def render(self):

        framebuffer = np.ascontiguousarray(self.framebuffer)

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
