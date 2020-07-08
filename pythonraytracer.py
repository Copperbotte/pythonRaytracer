
import matplotlib.pyplot as plt
import numpy as np
import random as rnd

class ray:
    def __init__(self):
        self.src = np.array([0,0,0])
        self.vec = np.array([0,0,0])

def length(vec):
    vec = np.array([vec])
    return np.sqrt(vec.dot(vec.transpose()))

def normalize(vec):
    return vec / length(vec)

def render(width=800, height=600):
    x = np.arange(width)
    y = np.arange(height)
    X = (2.0*x / width) - 1.0
    Y = (2.0*y / height) - 1.0
    Y *= height / width

    img = []
    for yPos in Y:
        line = []
        for xPos in X:
            Z = np.sin(np.pi * 2.0 * xPos) * np.sin(np.pi * 2.0 * yPos)
            line.append([Z,Z,Z])
        img.append(line)
    plt.imshow(img)
    plt.show()
    
    

    
