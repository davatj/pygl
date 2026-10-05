from .renderer.renderer import Renderer
from .scene.scene import Scene

def main():
    Renderer(Scene.load_scene("cube")).render()

if __name__ == "__main__":
    main()
