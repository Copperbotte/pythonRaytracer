
import matplotlib.pyplot as plt
import matplotlib.colors as mplcolor
import optical_tools as ot
import numpy as np
import random as rnd
import sys
import time

def vec3(i):
    i = float(i)
    return np.array([i,i,i])

class Ray:
    def __init__(self, s=vec3(0), v=vec3(0)):
        self.src = s
        self.vec = v

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

    epsilon = 1.0 - 1e-5
    if epsilon < cos:
        return vec
    if cos < -epsilon:
        return -vec;

    #make a pair of bases that are orthogonal to the normal
    b1 = np.cross(sampleNormal, normal)
    b1 = normalize(b1)
    b2 = np.cross(b1, normal)
    br = normalize(b2)
    
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

class tvHit:
    """traversal struct. Contains rIn, normal, id, and rayDist."""
    def __init__(self, ray=Ray(), norm=vec3(3), ID=0, RD=0.0):
        rIn = ray
        normal = norm
        Id = ID
        rayDist = RD

def tvSphere(rIn, hit, surfID, sRad, sPos):
    sOffset = sPos - rIn.src
    sRayClosest = dot(sOffset, rIn.vec)
    sRayClosestDist2 = dot(sOffset, sOffset) - sRayClosest**2
    sInnerOffset2 = sRad**2 - sRayClosestDist2
    if 0 < sInnerOffset2:
        #hit sphere, calculate intersection
        sInnerOffset = np.sqrt(sInnerOffset2)
        rDist = sRayClosest - sInnerOffset
        if rDist < hit.rayDist:
            hit.rayDist = rDist
            hit.rIn.src = rIn.src + rIn.vec * hit.rayDist
            hit.normal = normalize(rIn.src - sPos)
            hit.Id = surfID

def tvPlane(rIn, hit, surfID, normal, offset):
    #test if ray is above surface
    side = dot(normal, rIn.src)
    above = offset < side
    
    #test if ray points toward surface
    proj = dot(normal, rIn.vec)
    facing = proj < 0.0

    if above == facing:
        dst = Ray()
        dst.src = rIn.src + rIn.vec * (offset - side) / proj
        dst.vec = rIn.vec
        dstlen = length(dst.src - rIn.src)
        if dstlen < hit.rayDist:
            hit.rayDist = dstlen
            hit.rIn = dst
            hit.normal = normal
            hit.Id = surfID
            if not above:
                hit.normal = -normal

def traverse(scene, rIn, inID=0):
    hit = tvHit()
    hit.rIn = Ray()
    hit.rIn.src = rIn.src
    hit.rIn.vec = rIn.vec
    hit.normal = np.array([0,0,0])
    distMin = 0.001
    hit.rayDist = sys.float_info.max
    hit.Id = 0
    IDOffset = 1
    
    #floor
    for n in range(len(scene['planes'])):
        surfID = n + IDOffset
        plane = scene['planes'][n]
        norm = plane['surface'][0]
        offset = dot(norm, plane['surface'][1])
        if inID != surfID:
            tvPlane(rIn, hit, surfID, norm, offset)
    IDOffset += len(scene['planes'])
    
    #spheres
    for n in range(len(scene['spheres'])):
        surfID = n + IDOffset
        sphere = scene['spheres'][n]
        #project sphere's origin onto the ray
        sPos = sphere['surface'][0]
        sRad = sphere['surface'][1]
        if inID != surfID:
            tvSphere(rIn, hit, surfID, sRad, sPos)
    return hit       
    #return outRay, outNormal, outID

def lookupSceneByID(scene, ID):
    IDOffset = 1
    for Type in ['planes', 'spheres']:
        typeLen = len(scene[Type])
        if ID < typeLen + IDOffset:
            return scene[Type][ID - IDOffset]
        IDOffset += typeLen
    return None

#brdfs
def brdfLambert(hit, rOut):
    diffuse = dot(hit.normal, rOut.vec) / np.pi
    diffuse = max(0.0, diffuse)
    return diffuse

#samplers & their pdfs
def sampleSphere(hit):
    rOut = randSphere()
    if dot(rOut, hit.normal) < 0.0:
        rOut = reflect(rOut, hit.normal)
    pdf = 1.0 / (2.0 * np.pi)
    return rOut, pdf
def pdfSphere(hit, rOut):
    return 1.0 / (2.0 * np.pi)

def sampleLambert(hit):
    rOut = randLambert(hit.normal)
    pdf = brdfLambert(hit, Ray(v=rOut))
    return rOut, pdf
def pdfLambert(hit, rOut):
    return brdfLambert(hit, rOut)

def sampleLight(hit, sPos, sRad):
    #selects a random point to raytrace towards
    rPos = randSphere()
    rNorm = rPos.copy()

    #finds the vector from the ray source to this sample
    rPos = sRad*rPos + sPos - hit.rIn.src

    #pdf is the probability area projected from this point
    r2 = dot(rPos, rPos)
    rOut = rPos / np.sqrt(r2)

    area = 2.0 * np.pi * sRad**2
    area *= abs(dot(rNorm, rOut)) #projected area
    pdf = 1.0
    try:
        pdf = r2 / area
    except ZeroDivisionError as e:
        pdf = 0.0

    return rOut, pdf
def pdfLight(hit, rOut, sPos, sRad): #the big one
    delta = sPos - rOut.src
    ndelta = normalize(delta)
    delta2 = dot(delta, delta)
    r2 = sRad**2
    proj = dot(ndelta, rOut.vec)
    if proj < 0.0: #sphere is below ray
        return 0.0
    if delta2 < r2: #ray is within sphere
        return 0.0 #harder to code
    if proj**2 <= (1.0 - r2/delta2): #if the ray missed the sphere
        return 0.0

    #find intersection
    mdelta = np.sqrt(delta2)
    mmid = dot(delta, rOut.vec) # length of line that makes a right angle to the sphere
    mmid2 = mmid**2
    mdev2 = r2 + mmid2 - delta2
    if mdev2 < 0.0: #another missed sphere check
        return 0.0

    #mdev is the little length difference between mid and dist
    mdev = np.sqrt(mdev2)
    dist1 = mmid - mdev # is this a complex root?
    dist2 = mmid + mdev # a sqrt's involved, and there's two conjugates
    sHit1 = dist1 * rOut.vec + rOut.src
    sHit2 = dist2 * rOut.vec + rOut.src
    n1 = normalize(sHit1 - sPos)
    n2 = normalize(sHit2 - sPos)

    #find projected areas
    pdf = 0.0
    SA = 4.0 * np.pi * r2
    A1 = SA * abs(dot(n1, rOut.vec))
    A2 = SA * abs(dot(n2, rOut.vec))

    #area facing the ray, over the distance^2
    pdf1 = 0.0
    pdf2 = 0.0
    try:
        pdf1 = dist1**2 / A1
    except ZeroDivisionError as e:
        pdf1 = 0.0
    try:
        pdf2 = dist2**2 / A2
    except ZeroDivisionError as e:
        pdf2 = 0.0
    return pdf1 + pdf2

def sampleMIS(hit):
    #build method space
    c_Hemi = 1.0
    c_Lambert = 1.0
    c_Light = 1.0
    
    methodspace = [c_Hemi, c_Lambert, c_Light]
    methodspace = [m / sum(methodspace) for m in methodspace] #normalize sum
    
    #pick a method from methodspace
    pick = rnd.random()
    c = 0
    methodSum = 0.0
    for m in methodspace:
        methodSum += m
        if pick < methodSum:
            break
        c = c+1
    
    #find sample from chosen method
    rOut = Ray()
    rOut.src = hit.rIn.src
    
    pdf = 0.0
    if c == 0:
        rOut.vec, pdf = sampleSphere(hit)
    elif c == 1:
        rOut.vec, pdf = sampleLambert(hit)
    else:
        surf = scene['spheres'][0]['surface']
        rOut.vec, pdf = sampleLight(hit, surf[0], surf[1])

    #reduce probability by methodspace
    pdf *= methodspace[c]

    #find pdfs from other samplers, the heart of MIS
    for i in range(len(methodspace) - 1):
        j = (i+c+1) % len(methodspace) # skips the precomputed method
        if j == 0:
            pdf += methodspace[j] * pdfSphere(hit, rOut)
        elif j == 1:
            pdf += methodspace[j] * pdfLambert(hit, rOut)
        else: #j == 2:
            surf = scene['spheres'][0]['surface']
            pdf += methodspace[j] * pdfLight(hit, rOut, surf[0], surf[1])
    
    return rOut.vec, pdf

#raytracers
def raytrace_Hemisphere(scene, rIn, bounces, inID=0):
    hit = traverse(scene, rIn, inID)
    
    if hit.Id == 0:
        # hit sky, sky is not reflective
        return scene['sky']
    elif bounces == 0:
        surface = lookupSceneByID(scene, hit.Id)
        if surface == None:
            return np.array([0.0,0.0,0.0]) #error
        return surface['material']['emission']

    surface = lookupSceneByID(scene, hit.Id)
    color = surface['material']['albedo']
    emission = surface['material']['emission']

    #generate new ray
    rOut = Ray()
    rOut.src = hit.rIn.src
    rOut.vec, pdf = sampleSphere(hit)
    
    light = raytrace_Hemisphere(scene, rOut, bounces - 1, hit.Id)
    
    brdf = brdfLambert(hit, rOut)
    brdf = brdf * color
        
    return emission + light * brdf / pdf

def raytrace_Lambert(scene, rIn, bounces, inID=0):
    hit = traverse(scene, rIn, inID)
    
    if hit.Id == 0:
        # hit sky, sky is not reflective
        return scene['sky']
    elif bounces == 0:
        surface = lookupSceneByID(scene, hit.Id)
        if surface == None:
            return np.array([0.0,0.0,0.0]) #error
        return surface['material']['emission']

    surface = lookupSceneByID(scene, hit.Id)
    color = surface['material']['albedo']
    emission = surface['material']['emission']

    #generate new ray
    rOut = Ray()
    rOut.src = hit.rIn.src
    rOut.vec, pdf = sampleLambert(hit)
    
    light = raytrace_Lambert(scene, rOut, bounces - 1, hit.Id)
    
    brdf = brdfLambert(hit, rOut)
    brdf = brdf * color
        
    return emission + light * brdf / pdf

def raytrace_Light(scene, rIn, bounces, inID=0):
    hit = traverse(scene, rIn, inID)
    
    if hit.Id == 0:
        # hit sky, sky is not reflective
        return scene['sky']
    elif bounces == 0:
        surface = lookupSceneByID(scene, hit.Id)
        if surface == None:
            return np.array([0.0,0.0,0.0]) #error
        return surface['material']['emission']

    surface = lookupSceneByID(scene, hit.Id)
    color = surface['material']['albedo']
    emission = surface['material']['emission']

    #generate new ray
    rOut = Ray()
    rOut.src = hit.rIn.src
    surf = scene['spheres'][0]['surface']
    rOut.vec, pdf = sampleLight(hit, surf[0], surf[1])
    
    light = raytrace_Light(scene, rOut, bounces - 1, hit.Id)
    
    brdf = brdfLambert(hit, rOut)
    brdf = brdf * color
        
    return emission + light * brdf / pdf

def raytrace_Random_IS(scene, rIn, bounces, inID=0):
    hit = traverse(scene, rIn, inID)
    
    if hit.Id == 0:
        # hit sky, sky is not reflective
        return scene['sky']
    elif bounces == 0:
        surface = lookupSceneByID(scene, hit.Id)
        if surface == None:
            return np.array([0.0,0.0,0.0]) #error
        return surface['material']['emission']

    surface = lookupSceneByID(scene, hit.Id)
    color = surface['material']['albedo']
    emission = surface['material']['emission']

    #generate new ray
    rOut = Ray()
    rOut.src = hit.rIn.src

    pick = rnd.random()
    if pick < 1/3:
        rOut.vec, pdf = sampleSphere(hit)
    elif pick < 2/3:
        rOut.vec, pdf = sampleLambert(hit)
    else:
        surf = scene['spheres'][0]['surface']
        rOut.vec, pdf = sampleLight(hit, surf[0], surf[1])
    
    light = raytrace_Random_IS(scene, rOut, bounces - 1, hit.Id)
    
    brdf = brdfLambert(hit, rOut)
    brdf = brdf * color
        
    return emission + light * brdf / pdf

def raytrace_MIS(scene, rIn, bounces, inID=0):
    hit = traverse(scene, rIn, inID)
    
    if hit.Id == 0:
        # hit sky, sky is not reflective
        return scene['sky']
    elif bounces == 0:
        surface = lookupSceneByID(scene, hit.Id)
        if surface == None:
            return np.array([0.0,0.0,0.0]) #error
        return surface['material']['emission']

    surface = lookupSceneByID(scene, hit.Id)
    color = surface['material']['albedo']
    emission = surface['material']['emission']

    #generate new ray
    rOut = Ray()
    rOut.src = hit.rIn.src
    rOut.vec, pdf = sampleMIS(hit)
    
    light = raytrace_MIS(scene, rOut, bounces - 1, hit.Id)
    
    brdf = brdfLambert(hit, rOut)
    brdf = brdf * color
        
    return emission + light * brdf / pdf

def raytrace(scene, rIn, bounces, inID=0, mode="Hemisphere"):
    modes = {"Hemisphere":raytrace_Hemisphere,
             "SurfaceIS":raytrace_Lambert,
             "LightIS":raytrace_Light,
             "RandomIS":raytrace_Random_IS,
             "MIS":raytrace_MIS}
    return modes[mode](scene, rIn, bounces, inID)
    

def toTimeString(t):
    return '{0} minutes {1} seconds'.format(*divmod(t, 60))

def render(scene, width=800, height=600, samples=1, bounces=1, mode="Hemisphere"):
    #x = np.arange(width)
    #y = np.arange(height)
    #X = (2.0*x / width) - 1.0
    #Y = (2.0*y / height) - 1.0
    #Y *= height / width
    
    startTime = time.time()
    prevTime = startTime
    prevPercent = 0.0

    times = []
    pcts = []
    
    img = []
    #for yPos,yn in zip(Y[::-1], y):
    for y in range(height): #iterators like this seem to greatly speed up the render
        y = float(y)
        y = height - y - 1.0
        Y = (2.0*y / height) - 1.0
        Y *= height / width
        yPos = Y
        line = []
        #for xPos,xn in zip(X,x):
        for x in range(width):
            x = float(x)
            X = (2.0*x / width) - 1.0
            xPos = X
            curTime = time.time()
            #if 10.0 <= curTime - prevTime:
            #if prevTime != startTime:
            if 0.02 <= curTime - prevTime:
                elapsedTime = curTime - startTime
                dT = curTime - prevTime
                prevTime = curTime
                elapsedPercent = float((height-y)*width+x)/(float(width*height) / 100.0)
                pcts.append(elapsedPercent)
                times.append(elapsedTime)
                dP = elapsedPercent - prevPercent
                prevPercent = elapsedPercent
                r = dP/dT
                ir = dT/dP
                P0 = elapsedPercent - r*elapsedTime
                #print()
                #print(toTimeString(elapsedTime))
                #print(int(elapsedPercent), "% complete")
                ##print(toTimeString((100.0 / elapsedPercent - 1.0)*elapsedTime) + " remaining")
                #print(toTimeString((100.0-P0)*ir-elapsedTime) + " remaining")
            ray = Ray()
            ray.src = vec3(0)
            color = vec3(0)

            for i in range(samples):
                dx = 2.0*rnd.random() / width
                dy = 2.0*rnd.random() / height
                ray.vec = normalize(np.array([xPos + dx, 1.0, yPos + dy]))
                color += raytrace(scene, ray, bounces, mode=mode)
            line.append(ot.l2s(color / float(samples)))
        img.append(line)
        
    elapsedTime = time.time() - startTime
    print(toTimeString(elapsedTime) + ' total')
    fig = plt.figure(mode, figsize=(float(width)/100.0, float(height)/100.0))
    ax = fig.add_axes([0,0,1,1])
    ax.imshow(img)
    return times, pcts
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
    sRad = 1.0
    sPos = np.array([-1.0, 3.0, -1.0 + sRad])
    sphere['surface'] = [sPos, sRad]

    albedo = np.array([1.0,1.0,1.0])
    emission = np.array(ot.s2l([1.0,0.5,0.0]))
    sphere['material'] = {'albedo': albedo, 'emission': emission}

    spheres.append(sphere)

    #sphere array
    for i in range(10):
        aTime = 4.1 #+ iTime
        sphere = dict()
        sRad = 0.25
        sPos = np.array([1.0, 3.0, -1.0 + sRad])
        sPos += np.array([0.0,float(i)+1.0,0.0])
        sPos += np.array([2.0 * 0.3 * float(i+1)*np.cos(aTime + float(i) / 2.0),0.0,0.0])
        sphere['surface'] = [sPos, sRad]

        albedo = np.array([1.0,1.0,1.0])
        emission = np.array(ot.s2l([0.0,0.0,0.0]))
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
    #render(scene, width=x, height=y, samples=s, bounces=1)
    #render(scene, width=x, height=y, samples=s, bounces=1, mode="SurfaceIS")
    #render(scene, width=x, height=y, samples=s, bounces=1, mode="LightIS")
    #render(scene, width=x, height=y, samples=s, bounces=1, mode="RandomIS")
    times, pcts = render(scene, width=x, height=y, samples=s, bounces=1, mode="MIS")
    plt.show()
    plt.plot(times, pcts)
    plt.show()
    #testSampler()
    #testSampler(lambda: randLambert(np.array([0.0,1.0,0.0])))
    None
