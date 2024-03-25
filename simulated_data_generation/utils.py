import sys
sys.path.append('/mnt/storage/home/lchen6/anaconda3/envs/CIIGAN/lib/python3.7/site-packages/')
from mathutils import Vector, Matrix
import torch
import cv2

class Point():
    def __init__(self,x,y):
        self.x = x
        self.y = y


class Correspondence():
    def __init__(self,srcCoord,tgtCoord):
        self.srcCoord = srcCoord
        self.tgtCoord = tgtCoord

def dot( v1, v2 ):
    return v1[0]*v2[0]+v1[1]*v2[1]+v1[2]*v2[2]


def homogeneous_vec(point):
	return Vector((tuple(point[:]) + (1,)))

def project_vec(point):
	return point / point[-1]

def load_img_tensor(path,texPatchSize):
	img = cv2.imread(path)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img = cv2.resize(img,(texPatchSize,texPatchSize))
	img = torch.from_numpy(img)
	img = img.permute(2,0,1)
	return img

def save_img_tensor(img,path):
	img = img.permute(1,2,0)
	img = img.numpy()
	img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
	cv2.imwrite(path,img)
