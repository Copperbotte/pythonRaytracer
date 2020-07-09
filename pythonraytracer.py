
import matplotlib.pyplot as plt
import optical_tools as ot
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

def reflect(ray, normal):
    proj = dot(ray, normal)
    return ray - 2.0*proj*normal

def randSphere():
    Xi = [rnd.random(), rnd.random()]
    Xi = list(map(lambda x: x*2.0 - 1.0, Xi))
    cosphi = np.sqrt(1.0 - Xi[0]**2)
    theta = np.pi * 2.0 * Xi[1]
    x = cosphi * np.sin(theta)
    y = cosphi * np.cos(theta)
    return np.array([x, y, Xi[0]])

def traverse(scene, ray):
    outRay = Ray()
    outRay.src = None
    outRay.vec = ray.vec
    outNormal = np.array([0,0,0])
    distMin = 0.001
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
            if distMin < dstlen and dstlen < rayDist:
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
        if distMin < rDist and rDist < rayDist:
            rayDist = rDist
            outRay.src = ray.src + ray.vec * rayDist
            outNormal = normalize(outRay.src - sPos)
            
    return outRay, outNormal

def raytrace(scene, ray, bounces):
    outRay, normal = traverse(None, ray)
    if outRay.src is None:
        # hit sky, sky is not reflective
        return np.array(ot.s2l([0.0,0.0,0.1]))
    elif bounces == 0:
        if outRay.src[2] < -0.9: # floor
            return np.array([0.0,0.0,0.0])
        else: #light
            return np.array(ot.s2l([1.0,0.5,0.0]))
    
    #surface materials
    if outRay.src[2] < -0.9: # floor
        Z = np.sin(np.pi * 2.0 * outRay.src[0]) * np.sin(np.pi * 2.0 * outRay.src[1])
        if Z < 0.0: Z = 0.0
        Z = Z * 0.5 + 0.25
        color = np.array([Z,Z,Z])
        emission = np.array([0.0,0.0,0.0])
    else: #light
        color = np.array([1.0,1.0,1.0])
        emission = np.array(ot.s2l([1.0,0.5,0.0]))

    #generate new ray
    outRay.vec = randSphere()
    if dot(outRay.vec, normal) < 0.0:
        outRay.vec = reflect(outRay.vec, normal)

    light = raytrace(scene, outRay, bounces - 1)
    return light * color + emission

def render(width=800, height=600):
    x = np.arange(width)
    y = np.arange(height)
    X = (2.0*x / width) - 1.0
    Y = (2.0*y / height) - 1.0
    Y *= height / width

    samples = 16
    
    img = []
    for yPos in Y[::-1]:
        line = []
        for xPos in X:
            ray = Ray()
            ray.src = np.array([0,0,0])
            ray.vec = normalize(np.array([xPos, 1.0, yPos]))
            line.append(raytrace(None, ray, 1))

            for i in range(samples - 1):
                dx = 2.0*rnd.random() / width
                dy = 2.0*rnd.random() / height
                ray.vec = normalize(np.array([xPos + dx, 1.0, yPos + dy]))
                line[-1] += raytrace(None, ray, 1)
            line[-1] = ot.l2s(line[-1] / float(samples))
        img.append(line)
    plt.imshow(img)
    plt.show()
    
if __name__ == "__main__":
    render()

    
