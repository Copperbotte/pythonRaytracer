
import matplotlib.pyplot as plt
import numpy as np
import random as rnd
import sys

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
    rayDist = sys.float_info.max
    
    #floor
    norm = np.array([0,0,1])
    offset = dot(norm, np.array([0,0,-1]))
    if offset < dot(norm, ray.src):
        #above surface
        proj = dot(norm, ray.vec)
        if proj < 0:
            dst = Ray()
            dst.src = ray.src - ray.vec / proj
            
            dstlen = length(dst.src - ray.src)
            if dstlen < rayDist:
                None
                rayDist = dstlen
                outRay = dst
                
    #spheres
    #project sphere's origin onto the ray
    sPos = np.array([-0.25, 2.0, 0.0])
    sRad = 0.3333
    
    sOffset = sPos - ray.src
    sRayClosest = dot(sOffset, ray.vec)
    sRayClosestDist2 = dot(sOffset, sOffset) - sRayClosest**2
    sInnerOffset2 = sRad**2 - sRayClosestDist2
    if 0 < sInnerOffset2:
        #hit sphere, calculate intersection
        sInnerOffset = np.sqrt(sInnerOffset2)
        rDist = sRayClosest - sInnerOffset
        if rDist < rayDist:
            rayDist = rDist
            outRay.src = ray.src + ray.vec * rayDist
            
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
            if outRay.src[2] < -0.9:
                Z = np.sin(np.pi * 2.0 * outRay.src[0]) * np.sin(np.pi * 2.0 * outRay.src[1])
                line.append([Z,Z,Z])
            else:
                line.append([1.0,0.5,0.0])
            
        img.append(line)
    plt.imshow(img)
    plt.show()
    
if __name__ == "__main__":
    render()

    
