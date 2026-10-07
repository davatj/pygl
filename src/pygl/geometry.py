import numpy as np

def orient2d(a: np.ndarray[tuple[int], float], 
             b: np.ndarray[tuple[int], float], 
             c: np.ndarray[tuple[int], float]):

    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])