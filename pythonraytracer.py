
import matplotlib.pyplot as plt
import numpy as np
import random as rnd

class Ray:
    def __init__(self):
        self.src = np.array([0,0,0])
        self.vec = np.array([0,0,0])

def dot(v1, v2):
    v1 = np.array([v1])
    v2 = np.array([v2])
    return v1.dot(v2.transpose())[0][0]

def length(vec):
    return np.sqrt(dot(vec, vec))

def normalize(vec):
    return vec / length(vec)

def traverse(scene, ray):
    outRay = Ray()
    outRay.src = None
    outRay.vec = ray.vec
    
    #floor
    norm = np.array([0,0,1])
    offset = -1
    if offset < dot(norm, ray.src):
        #above surface
        proj = dot(norm, ray.vec)
        if proj < 0:
            outRay.src = ray.src - ray.vec / proj
    
    return outRay

def render(width=800, height=600):
    x = np.arange(width)
    y = np.arange(height)
    X = (2.0*x / width) - 1.0
    Y = (2.0*y / height) - 1.0
    Y *= height / width
    
    img = []
    for yPos in Y[::-1]:
        line = []
        for xPos in X:
            ray = Ray()
            ray.src = np.array([0,0,0])
            ray.vec = normalize(np.array([xPos, 1.0, yPos]))

            outRay = traverse(None, ray)
            if outRay.src is None:
                line.append([0.0,0.0,0.0])
                continue

            Z = np.sin(np.pi * 2.0 * outRay.src[0]) * np.sin(np.pi * 2.0 * outRay.src[1])
            line.append([Z,Z,Z])
        img.append(line)
    plt.imshow(img)
    plt.show()
    
if __name__ == "__main__":
    render()

    
