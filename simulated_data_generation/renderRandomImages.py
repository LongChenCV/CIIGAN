import argparse
import sys
sys.path.append('/mnt/storage/home/lchen6/anaconda3/envs/CIIGAN/lib/python3.7/site-packages/')
import bpy
import random
from mathutils import Vector, noise
import traceback
import numpy as np
print('Hello, already synchronized between local and remote server')

# Add path to this file to the sys path to be able to import local files:
blenderFilePath = bpy.path.abspath("//")
sys.path.append(blenderFilePath)

from raycastTensified import Raycaster

# Remove blender-related arguments:
argv = sys.argv
try:
    argv = argv[argv.index("--") + 1:]  # get all args after "--"
except:
    argv = []

parser = argparse.ArgumentParser(description="Randomly position camera and render frame")
parser.add_argument("--images", type=int, default=1, help="Number of random images to render")
parser.add_argument("--start_no", type=int, default=0, help="continue at image number X")
parser.add_argument("--img_width", type=int, default=512, help="")
parser.add_argument("--img_height", type=int, default=256, help="")
parser.add_argument("--texture_patch_size", type=int, default=512, help="Size of texture patch from one direction (width and height)")
parser.add_argument("--test_render", action="store_true", help="Render and save an image for every random camera pose")
parser.add_argument("--triplets", action="store_true", help="generate 3 Images with small, random, linear motion")

args = parser.parse_args(argv)

scn = bpy.context.scene
cam = bpy.data.objects["Camera"]


# From Kosvor/zeffii on https://github.com/nortikin/sverchok/issues/660:
def isInsideMesh(p, obj):
    pObj = obj.matrix_world.inverted()@p
    if bpy.app.version >= (2, 79, 0):    # API changed
        res, point, normal, face = obj.closest_point_on_mesh(pObj)
    else:
        point, normal, face = obj.closest_point_on_mesh(pObj)
    p2 = point-pObj
    #print(p2.length)
    #if p2.length < min_dist:
    #    return False
    v = p2.dot(normal)
    #print("v",v)
    #print(obj)
    #indicator.location = obj.matrix_world*pObj
    #print("p", point, obj.matrix_world*point)
    return (v >= 0.0)

# From aothms on https://blenderartists.org/t/detecting-if-a-point-is-inside-a-mesh-2-5-api/485866/4
def pointInsideMesh(point,ob):
    axes = [Vector((1, 0, 0)), Vector((0, 1, 0)), Vector((0, 0, 1))]
    outside = False

    mat = ob.matrix_world.inverted()
    #f = obj.ray_cast(mat * ray_origin, mat * ray_destination)
    for axis in axes:
        orig = mat@point
        count = 0
        while True:
            if bpy.app.version >= (2, 77, 0):    # API changed
                result,location,normal,index = ob.ray_cast(orig,axis)
            else:
                end = orig+axis*1e10
                location,normal,index = ob.ray_cast(orig,end)
            if index == -1: break
            count += 1
            orig = location + axis*0.01
        if count%2 == 0:
            outside = True
            break
    return not outside

def distToObject(p, obj):
    pObj = obj.matrix_world.inverted()@p
    if (2, 79, 0) <= bpy.app.version:    # API changed
        res, point, normal, face = obj.closest_point_on_mesh(pObj)
    else:
        point, normal, face = obj.closest_point_on_mesh(pObj)
    dist = (obj.matrix_world@point - p).length
    return dist

# Converts bounding box to min, max format:
def getObjExtent( obj ):
    bb = obj.bound_box
    minX, minY, minZ = float('Inf'),float('Inf'),float('Inf')
    maxX, maxY, maxZ = -float('Inf'), -float('Inf'), -float('Inf')
    for v in obj.bound_box:
        if v[0] < minX:
            minX = v[0]
        if v[1] < minY:
            minY = v[1]
        if v[2] < minZ:
            minZ = v[2]
        if v[0] > maxX:
            maxX = v[0]
        if v[1] > maxY:
            maxY = v[1]
        if v[2] > maxZ:
            maxZ = v[2]
    return minX, maxX, minY, maxY, minZ, maxZ

# Retrieve a random point inside a mesh:
def getRandomPointIn(obj, inside=[], outside=[], maxTries=1000, distToWall=10):
    tries = 0
    minX, maxX, minY, maxY, minZ, maxZ = getObjExtent(obj)
    while tries < maxTries:
        x = random.uniform(minX, maxX)
        y = random.uniform(minY, maxY)
        z = random.uniform(minZ, maxZ)
        p = Vector((x, y, z))
        p = obj.matrix_world@p
        valid = False
        if pointInsideMesh(p, obj):# and distToObject( p, obj ) > distToWall:
            valid = True
        if valid:
            for i in inside:
                if not pointInsideMesh(p, i):# or distToObject( p, i ) < distToWall:
                    valid = False
                    break
        if valid:
            for o in outside:
                #print( "outside check", o )
                if pointInsideMesh(p, o):#or distToObject( p, o ) < distToWall:
                    #print( "\tinside" )
                    valid = False
                    break
                #print( "\toutside" )
        if valid:
            #print( "Found random position after " + str(tries) + " tries." )
            return p
        tries += 1
    raise ValueError("Could not find random point in object. Giving up." )

camPositionVolume = bpy.data.objects["CamPositionVolume"]
camPositionVolume.hide_render = True        # Don't render this object

# Compose a list of meshes which the camera must not penetrate:
# If a mesh isn't found, ignore it.
impenetrable = []
for name in ["Liver", "Gallbladder", "Fat", "Ligament"]:
    try:
        obj = bpy.data.objects[name]
        impenetrable.append(obj)
    except:
        pass
posInside = [bpy.data.objects["AbdominalWall"]]

liver = bpy.data.objects["Liver"]
cam = bpy.data.objects["Camera"]
lookTarget = bpy.data.objects["LookTarget"]

sceneName = bpy.path.basename(bpy.context.blend_data.filepath)
#print(sceneName)
raycaster = Raycaster( sceneName[:-6], args.img_width, args.img_height, args.texture_patch_size, args.texture_patch_size )

for i in range( args.images - args.start_no ):

    if args.triplets:
        i = (i*3) + 1
        #i = (i*2) + 1

    i += args.start_no
    print( "Image:", i )
    
    # Place camera randomly:
    try:
        camPos = getRandomPointIn( camPositionVolume, outside=impenetrable, maxTries = 9999 )
        lookPosOutside = [bpy.data.objects["Fat"]]
        camLookPos = getRandomPointIn( liver, outside=lookPosOutside,
                maxTries = 9999, distToWall=0 )
    except Exception as e:
        print("Failed to find valid pose for image {}, skipping".format( i ))
        traceback.print_exc()
        continue

    # Always let the camera look "upwards" towards the patient's head:
    if camLookPos.y <= camPos.y:
        camLookPos.y = camPos.y + 30
   
    cam.location = camPos
    lookTarget.location = camLookPos

    scn = bpy.context.scene
    bpy.context.view_layer.update()

    # Run raycasting and saVe the found texture <-> image correspondences to a file:
    raycaster.raycast(i, generate_sequence=False)

    # Optionally render the image from the current camera pose
    # (mainly for debugging purposes or image-based baselines)
    # (our model does not use these. reference images are rendered at train time)
    if args.test_render:
        raycaster.testRender(i)

    # the ReCycle-GAN baselines were tzrained on triplets of consecutive video frames
    # see "http://www.cs.cmu.edu/~aayushb/Recycle-GAN/"
    # to recreate this training setting, we generate synthetic 3-frame sequences
    if args.triplets:

        delta_cam = Vector(np.random.normal(size=3))
        delta_look = Vector(np.random.normal(size=3))

        acc_cam = np.random.normal(loc=1)
        acc_look = np.random.normal(loc=1)
        
        i_prev = i-1 
        cam.location -= delta_cam
        lookTarget.location -= delta_look
        scn = bpy.context.scene
        bpy.context.view_layer.update()
        raycaster.raycast(i_prev, generate_sequence=False)
        if args.test_render:
            raycaster.testRender(i_prev)

        i_succ = i+1 
        cam.location += delta_cam + acc_cam*delta_cam
        lookTarget.location += delta_look + acc_look*delta_look
        scn = bpy.context.scene
        bpy.context.view_layer.update()
        raycaster.raycast(i_succ, generate_sequence=False)
        if args.test_render:
            raycaster.testRender(i_succ)