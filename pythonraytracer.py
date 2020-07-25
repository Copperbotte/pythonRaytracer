
import matplotlib.pyplot as plt
import matplotlib.colors as mplcolor
import optical_tools as ot
import numpy as np
import random as rnd
import sys
import time

class Ray:
    def __init__(self):
        self.src = np.array([0,0,0])
        self.vec = np.array([0,0,0])

def clamp(v, low=0.0, high=1.0):
    if v < low:
        return low
    if high < v:
        return high
    return v

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
    Xi[0] = Xi[0] * 2.0 - 1.0
    cospsi = np.sqrt(1.0 - Xi[0]**2)
    theta = np.pi * 2.0 * Xi[1]
    x = cospsi * np.cos(theta)
    y = cospsi * np.sin(theta)
    z = Xi[0]
    return np.array([x, y, z])

def toWorld(vec, normal):
    #todo: do this properly using quaternions
    sampleNormal = np.array([0.0,0.0,1.0])
    cos = dot(sampleNormal, normal)
    if 0.99 < cos:
        return vec

    #make a pair of bases that are orthogonal to the normal
    b1 = np.cross(sampleNormal, normal)
    b1 = normalize(b1)
    b2 = np.cross(b1, normal)

    #rotate toward normal
    M = np.array([b1, b2, normal]).transpose()

    return M.dot(vec)

def randLambert(normal):
    Xi = [rnd.random(), rnd.random()]
    sinpsi = np.sqrt(Xi[0])
    cospsi = np.sqrt(1.0 - sinpsi**2)
    theta = np.pi * 2.0 * Xi[1]
    x = cospsi * np.cos(theta)
    y = cospsi * np.sin(theta)
    z = sinpsi

    vec = np.array([x, y, z])
    #rotate toward normal
    
    return toWorld(np.array([x, y, z]), normal)

def randLight(scene):
    #currently the only lights are spheres.
    surface = scene['spheres'][0]
    sPos = surface['surface'][0]
    rPos = randSphere() * surface['surface'][1] + sPos
    pdf = 1.0 / (4.0 * np.pi * surface['surface'][1]**2) #sphere total area
    return rPos, normalize(rPos - sPos), pdf

def traverse(scene, ray, inID=0):
    outRay = Ray()
    outRay.src = None
    outRay.vec = ray.vec
    outNormal = np.array([0,0,0])
    distMin = 0.001
    rayDist = sys.float_info.max
    outID = 0
    IDOffset = 1
    
    #floor
    for n in range(len(scene['planes'])):
        surfID = n + IDOffset
        plane = scene['planes'][n]
        norm = plane['surface'][0]
        offset = dot(norm, plane['surface'][1])
        if offset < dot(norm, ray.src):
            #above surface
            proj = dot(norm, ray.vec)
            if proj < 0:
                dst = Ray()
                dst.src = ray.src - ray.vec / proj
                
                dstlen = length(dst.src - ray.src)
                if inID != surfID and dstlen < rayDist:
                    rayDist = dstlen
                    outRay = dst
                    outNormal = norm
                    outID = surfID
    IDOffset += len(scene['planes'])
    
    #spheres
    for n in range(len(scene['spheres'])):
        surfID = n + IDOffset
        sphere = scene['spheres'][n]
        #project sphere's origin onto the ray
        sPos = sphere['surface'][0]
        sRad = sphere['surface'][1]
        sOffset = sPos - ray.src
        sRayClosest = dot(sOffset, ray.vec)
        sRayClosestDist2 = dot(sOffset, sOffset) - sRayClosest**2
        sInnerOffset2 = sRad**2 - sRayClosestDist2
        if 0 < sInnerOffset2:
            #hit sphere, calculate intersection
            sInnerOffset = np.sqrt(sInnerOffset2)
            rDist = sRayClosest - sInnerOffset
            if inID != surfID and rDist < rayDist:
                rayDist = rDist
                outRay.src = ray.src + ray.vec * rayDist
                outNormal = normalize(outRay.src - sPos)
                outID = surfID
            
    return outRay, outNormal, outID

def lookupSceneByID(scene, ID):
    IDOffset = 1
    for Type in ['planes', 'spheres']:
        typeLen = len(scene[Type])
        if ID < typeLen + IDOffset:
            return scene[Type][ID - IDOffset]
        IDOffset += typeLen
    return None

def raytrace(scene, ray, bounces, inID=0, mode="Hemisphere"):
    outRay, normal, outID = traverse(scene, ray, inID)
    if outRay.src is None:
        # hit sky, sky is not reflective
        return scene['sky']
    elif bounces == 0:
        surface = lookupSceneByID(scene, outID)
        if surface == None:
            return np.array([0.0,0.0,0.0]) #error
        return surface['material']['emission']
    
    #surface materials
    #if outID == 1: # floor
    #    #Z = np.sin(np.pi * 2.0 * outRay.src[0]) * np.sin(np.pi * 2.0 * outRay.src[1])
    #    Z = 1.0
    #    Z = clamp(Z)
    #    #Z = Z * 0.5 + 0.25
    #    color = np.array([Z,Z,Z])
    #    emission = np.array([0.0,0.0,0.0])
    #else: #light
    #    color = np.array([1.0,1.0,1.0])
    #    emission = np.array(ot.s2l([1.0,0.5,0.0]))

    surface = lookupSceneByID(scene, outID)
    color = surface['material']['albedo']
    emission = surface['material']['emission']
    
    #generate new ray
    hemipdf = 1.0 / (2.0 * np.pi)
    samplemode = mode

    def hemisphere():
        return randSphere(), 1.0 / np.pi
    
    def surfaceIS(normal):
        vec = randLambert(normal)
        pdf = hemipdf # uniform pdf
        return vec, pdf

    def lightIS(curPos):
        rPos, lnorm, pdf = randLight(scene)
        rDiff = rPos - curPos
        vec = normalize(rDiff)
        #samplepdf maps the probability area from one area to another
        #this *projects* the light differential area onto the sphere
        #pdf = 1.0 / (4.0 * np.pi * 0.3333**2)#sphere total area
        pdf *= abs(dot(lnorm, vec))          #area projection
        pdf /= dot(rDiff, rDiff)             #area scale
        pdf *= hemipdf                       #scale to final pdf
        #todo: rederive light IS to fix this bug
        return vec, pdf
    
    if mode == "SurfaceIS":
        outRay.vec, samplepdf = surfaceIS(normal)
    elif mode == "LightIS":
        outRay.vec, samplepdf = lightIS(outRay.src)
    elif mode == "MIS":
        lvec, lpdf = lightIS(outRay.src)
        svec, spdf = surfaceIS(normal)
        
        #MIS single sample
        lpdfWeight = lpdf / (lpdf + spdf)
        
        if rnd.random() < lpdfWeight:
            samplemode = "LightIS"
            samplepdf = lpdf * lpdfWeight
            outRay.vec = lvec
        else:
            samplemode = "SurfaceIS"
            samplepdf = spdf * (1.0 - lpdfWeight)
            outRay.vec = svec
    else:
        outRay.vec = randSphere()
        samplepdf = 1.0 / np.pi

    #"clamp"
    if dot(outRay.vec, normal) < 0.0:
        outRay.vec = reflect(outRay.vec, normal)
    
    light = raytrace(scene, outRay, bounces - 1, outID)
    
    if samplemode == "SurfaceIS":
        diffuse = 1.0
    elif samplemode == "LightIS":
        diffuse = dot(outRay.vec, normal)
    else:
        diffuse = dot(outRay.vec, normal)
        
    return emission + light * color * diffuse * (samplepdf / hemipdf)

def toTimeString(t):
    return '{0} minutes {1} seconds'.format(*divmod(t, 60))

def render(scene, width=800, height=600, samples=1, bounces=1, mode="Hemisphere"):
    x = np.arange(width)
    y = np.arange(height)
    X = (2.0*x / width) - 1.0
    Y = (2.0*y / height) - 1.0
    Y *= height / width
    
    startTime = time.time()
    prevTime = startTime
    
    img = []
    for yPos,yn in zip(Y[::-1], y):
        line = []
        for xPos,xn in zip(X,x):
            curTime = time.time()
            if 10.0 <= curTime - prevTime:
                elapsedTime = curTime - startTime
                prevTime = curTime
                elapsedPercent = float(yn*width+xn)/(float(width*height) / 100.0)
                print()
                print(toTimeString(elapsedTime))
                print(int(elapsedPercent), "% complete")
                print(toTimeString((100.0 / elapsedPercent - 1.0)*elapsedTime) + " remaining")
            ray = Ray()
            ray.src = np.array([0,0,0])
            ray.vec = normalize(np.array([xPos, 1.0, yPos]))
            line.append(raytrace(scene, ray, bounces, mode=mode))
            
            for i in range(samples - 1):
                dx = 2.0*rnd.random() / width
                dy = 2.0*rnd.random() / height
                ray.vec = normalize(np.array([xPos + dx, 1.0, yPos + dy]))
                line[-1] += raytrace(scene, ray, bounces, mode=mode)
            line[-1] = ot.l2s(line[-1] / float(samples))
        img.append(line)
        
    elapsedTime = time.time() - startTime
    print(toTimeString(elapsedTime) + ' total')
    fig = plt.figure(mode, figsize=(float(width)/100.0, float(height)/100.0))
    ax = fig.add_axes([0,0,1,1])
    ax.imshow(img)
    #plt.show()

def testSampler(sampler=randSphere):
    highest = 0.0
    img = [[0.0 for x in range(800)] for y in range(600)]
    for n in range(100000):
        r = sampler()
        r *= 300
        #r = np.array([r[0], r[2], r[1]])
        r *= np.array([1.0,-1.0,1.0])
        r += np.array([400.0,300.0,0.0])
        x = int(r[0])
        y = int(r[1])
        try:
            img[y][x] += 1.0
        except Exception as e:
            print(e, x, y)
        if highest < img[y][x]:
            highest = img[y][x]
    
    for y in range(len(img)):
        for x in range(len(img[0])):
            img[y][x] /= highest
            hsv = mplcolor.rgb_to_hsv(img[y][x])
            if 1.0 < hsv[2]:
                hsv[1] /= hsv[2]
                hsv[2] = 1.0
            img[y][x] = mplcolor.hsv_to_rgb(hsv)
            
    plt.imshow(img)
    plt.show()

def defaultScene():
    #no bsp or bvh for now
    scene = dict()
    scene['sky'] = np.array(ot.s2l([0.0,0.0,0.1]))
    
    #planes
    planes = []

    plane = dict()
    norm = np.array([0,0,1])
    point = np.array([0,0,-1])
    plane['surface'] = [norm, point]
    
    albedo = np.array([1.0,1.0,1.0]) # todo: procedural textures
    emission = np.array([0.0,0.0,0.0])
    plane['material'] = {'albedo': albedo, 'emission': emission}
    
    planes.append(plane)
    
    scene['planes'] = planes

    #spheres
    spheres = []

    sphere = dict()
    sRad = 0.3333
    sPos = np.array([-0.25, 2.0, -1.0 + sRad])
    sphere['surface'] = [sPos, sRad]

    albedo = np.array([1.0,1.0,1.0])
    emission = np.array(ot.s2l([1.0,0.5,0.0]))
    sphere['material'] = {'albedo': albedo, 'emission': emission}

    spheres.append(sphere)
    scene['spheres'] = spheres

    for key in scene.keys():
        print(scene[key])
    
    return scene
    

if __name__ == "__main__":
    scene = defaultScene()
    x = 800
    y = 600
    s = 1
    render(scene, width=x, height=y, samples=s, bounces=1)
    render(scene, width=x, height=y, samples=s, bounces=1, mode="SurfaceIS")
    render(scene, width=x, height=y, samples=s, bounces=1, mode="LightIS")
    render(scene, width=x, height=y, samples=s, bounces=1, mode="MIS")
    plt.show()
    #testSampler()
    #testSampler(lambda: randLambert(np.array([0.0,1.0,0.0])))
    None
