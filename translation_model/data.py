"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import torch.utils.data as data
import os.path
import random, math
from torchvision import transforms
import torchvision
from collections import namedtuple
import torch
import numpy

def default_loader(path):
    return Image.open(path).convert('RGB')


def default_flist_reader(flist):
    """
    flist format: impath label\nimpath label\n ...(same to caffe's filelist)
    """
    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            impath = line.strip()
            imlist.append(impath)

    return imlist


class ImageFilelist(data.Dataset):
    def __init__(self, root, flist, transform=None,
                 flist_reader=default_flist_reader, loader=default_loader):
        self.root = root
        self.imlist = flist_reader(flist)
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        impath = self.imlist[index]
        img = self.loader(os.path.join(self.root, impath))
        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.imlist)


class ImageLabelFilelist(data.Dataset):
    def __init__(self, root, flist, transform=None,
                 flist_reader=default_flist_reader, loader=default_loader):
        self.root = root
        self.imlist = flist_reader(os.path.join(self.root, flist))
        self.transform = transform
        self.loader = loader
        self.classes = sorted(list(set([path.split('/')[0] for path in self.imlist])))
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        self.imgs = [(impath, self.class_to_idx[impath.split('/')[0]]) for impath in self.imlist]

    def __getitem__(self, index):
        impath, label = self.imgs[index]
        img = self.loader(os.path.join(self.root, impath))
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)

###############################################################################
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code so that it also loads images from the current
# directory as well as the subdirectories
###############################################################################

import torch.utils.data as data

from PIL import Image, ImageOps, ImageDraw
import numpy
import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

RENDERDATA_EXTENSIONS = [
    '.tar'
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def is_render_file(filename):
    return any(filename.endswith(extension) for extension in RENDERDATA_EXTENSIONS)


def make_dataset(dir, render_data=False):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):        
        if os.path.basename(root) in ['2019_02_14','2019_10_10','2020_10_06']:
            print('skipped {}'.format(os.path.basename(root)))
            continue
        for fname in fnames:
            if not root.endswith( "ids" ) and not root.endswith( "labels" ) and not root.endswith( "normals" ) and not root.endswith( "depths" ):
                if not render_data:
                    if is_image_file(fname):
                        path = os.path.join(root, fname)
                        images.append(path)
                else:
                    if is_render_file(fname):
                        path = os.path.join(root, fname)
                        images.append(path)

    return images


def circleMask( img, ox, oy, radius ):
    mask = Image.new('L', img.size, 255)
    draw = ImageDraw.Draw(mask)
    x0 = img.size[0]*0.5 - radius + ox
    x1 = img.size[0]*0.5 + radius + ox
    y0 = img.size[1]*0.5 - radius + oy
    y1 = img.size[1]*0.5 + radius + oy
    draw.ellipse([x0,y0,x1,y1], fill=0)
    img.paste( (0,0,0), mask=mask )
    return img

class ImageFolder(data.Dataset):

    def __init__(self, root, return_labels=False,
            loader=default_loader, return_paths=False,
            augmentation={}):
        imgs = sorted(make_dataset(root))
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.return_paths = return_paths
        self.loader = loader
        self.return_labels = return_labels
        self.output_size = augmentation["output_size"]
        self.add_circle_mask = "circle_mask" in augmentation and augmentation["circle_mask"] == True
        self.rotate = "rotate" in augmentation and augmentation["rotate"] == True
        self.contrast = "contrast" in augmentation and augmentation["contrast"] == True
        if "new_size_min" in augmentation and "new_size_max" in augmentation:
            self.new_size_min = augmentation["new_size_min"]
            self.new_size_max = augmentation["new_size_max"]
        else:
            self.new_size_min = min(self.output_size)
            self.new_size_max = min(self.output_size)

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)

        maskRadius, maskOx, maskOy = None, None, None

        randAng = random.random()*20-10
        minOutputSize = min(self.output_size)
        maxOutputSize = max(self.output_size)
        aspect_ratio = img.height / img.width
        randSize = random.randint( self.new_size_min, self.new_size_max )
        randW = random.randint( self.new_size_min, self.new_size_max )
        randH = int(randW * aspect_ratio)

        if self.add_circle_mask:
            minSize = min((img.width, img.height))
            maxSize = max((img.width, img.height))
            maxRadius = int(math.sqrt((img.width/2)**2 + (img.height/2)**2))
            minRadius = int(0.4*maxSize)
            maskRadius = random.randint( minRadius, maxRadius )
            maskOx = random.randint( int(-img.width*0.1), int(img.width*0.1) )
            maskOy = random.randint( int(-img.height*0.1), int(img.height*0.1) )
            #maxRadius = int(math.sqrt((maxOutputSize/2)**2 + (minOutputSize/2)**2)*1.5)
            #minRadius = int(3*minOutputSize/4)
            #maskRadius = random.randint( minRadius, maxRadius )
            #maskOx = random.randint( int(-self.output_size[0]*0.3), int(self.output_size[0]*0.3) )
            #maskOy = random.randint( int(-self.output_size[1]*0.3), int(self.output_size[1]*0.3) )
            img = circleMask( img, maskOx, maskOy, maskRadius )

        #img = transforms.functional.resize( img, randSize, Image.BILINEAR )
        img = transforms.functional.resize( img, (randH,randW), Image.BILINEAR )
        if self.rotate:
            img = transforms.functional.rotate( img, randAng, Image.BILINEAR )
        
        rx = random.randint( 0, max(img.width - self.output_size[0],0) )
        ry = random.randint( 0, max(img.height - self.output_size[1],0) )
        img = transforms.functional.crop( img, ry, rx, self.output_size[1], self.output_size[0] )
            #img.save("circle.png")
        img = transforms.functional.to_tensor( img )
        if self.contrast:
            c = random.uniform( 0.75, 1.25 )
            b = random.uniform( -0.1, 0.1 )
            img = img*c + b

        img = transforms.functional.normalize( img, (0.5,0.5,0.5), (0.5,0.5,0.5) )


        #img = img + torch.randn_like( img )*0.1

        if self.return_labels:
            filePath, fileName = os.path.split(path)
            basePath, _ = os.path.split( filePath )
            labelPath = os.path.join( basePath, "labels", "lbl" + fileName[3:] )
            lbl = self.loader(labelPath)
            if self.add_circle_mask:
                lbl = circleMask( lbl, maskOx, maskOy, maskRadius )
            #lbl = transforms.functional.resize( lbl, randSize, Image.NEAREST )
            lbl = transforms.functional.resize( lbl, (randH,randW), Image.NEAREST )
            if self.rotate:
                lbl = transforms.functional.rotate( lbl, randAng, Image.NEAREST )
            lbl = transforms.functional.crop( lbl, ry, rx, self.output_size[1], self.output_size[0] )
                #lbl.save("lbl.png")
            lbl = torch.Tensor( numpy.asarray(lbl).transpose( 2,0,1 ) )

            #torchvision.utils.save_image( lbl.type(torch.FloatTensor)*255/6, "lbl.png", normalize=True)
            lbl = label2SingleChannel( lbl )

            #torchvision.utils.save_image( img, "img.png", normalize=True )

            #torchvision.utils.save_image( lbl.type(torch.FloatTensor)*255/6, "lbl2.png", normalize=True)

            if self.return_paths:
                return img, lbl, path
            else:
                return img, lbl
        else:
            if self.return_paths:
                return img, path
            else:
                return img

    def __len__(self):
        return len(self.imgs)

class RenderFolder(data.Dataset):

    def __init__(self, root, scene_to_index, translation_mode=False):
        self.imgs = sorted(make_dataset(root, render_data=True))
        if len(self.imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(RENDERDATA_EXTENSIONS)))
        self.scene_to_index = scene_to_index
        self.translation_mode = translation_mode

    def __getitem__(self, index):
        path = self.imgs[index]
        imgpath = path.replace('.tar', '.png')
        imgpath = imgpath.replace('corrData', 'cii')
        # Source code of read the CII or LBP images
        lbpimg=Image.open(imgpath).convert('RGB')
        lbpimg = transforms.functional.to_tensor(lbpimg)
        lbpimg = transforms.functional.normalize(lbpimg, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        renderData = torch.load(path)
        indTexList = [t.type(torch.LongTensor).cuda() for t in renderData['indTexList']]
        indImList = [i.type(torch.LongTensor).cuda() for i in renderData['indImList']]
        weights = renderData['weights'].type(torch.FloatTensor).cuda()
        depth = renderData['depth'].type(torch.FloatTensor).cuda().unsqueeze(dim=0)
        diffusion = renderData['diffusion'].type(torch.FloatTensor).cuda().unsqueeze(dim=0)
        projectionMatrix = renderData['projection_matrix'].cuda().unsqueeze(dim=0)
        points3D = renderData['points3D'].cuda()

        if not self.translation_mode:
            scene_dir = '/'.join(path.split('/')[:-1])
            scene_views = [f for f in os.listdir(scene_dir) if '.tar' in f and f != self.imgs[index].split('/')[-1]]
            new_view = scene_views[random.randint(0, len(scene_views)-1)]
            current_path=os.path.join(scene_dir, new_view)
            current_path = current_path.replace('.tar', '.png')
            current_path = current_path.replace('corrData', 'im')
            lbpimg_2 = Image.open(current_path).convert('RGB')
            lbpimg_2 = transforms.functional.to_tensor(lbpimg_2)
            lbpimg_2 = transforms.functional.normalize(lbpimg_2, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

            renderData_2 = torch.load(os.path.join(scene_dir, new_view))
            indTexList_2 = [t.type(torch.LongTensor).cuda() for t in renderData_2['indTexList']]
            indImList_2 = [i.type(torch.LongTensor).cuda() for i in renderData_2['indImList']]
            weights_2 = renderData_2['weights'].type(torch.FloatTensor).cuda()
            depth_2 = renderData_2['depth'].type(torch.FloatTensor).cuda().unsqueeze(dim=0)
            diffusion_2 = renderData_2['diffusion'].type(torch.FloatTensor).cuda().unsqueeze(dim=0)
            projectionMatrix_2 = renderData_2['projection_matrix'].cuda().unsqueeze(dim=0)
            points3D_2 = renderData_2['points3D'].cuda()

        scene = path.split('/')[-2]
        scene_idx = self.scene_to_index[scene]
        
        if not self.translation_mode:
            return (indTexList, indImList, weights, depth, diffusion, projectionMatrix, points3D, lbpimg), \
                    (indTexList_2, indImList_2, weights_2, depth_2, diffusion_2, projectionMatrix_2, points3D_2, lbpimg_2), \
                    scene_idx
        else:
            return (indTexList, indImList, weights, depth, diffusion, projectionMatrix, points3D, lbpimg), \
                    scene_idx, \
                    path


    def __len__(self):
        return len(self.imgs)

##########################################################
## One-Hot encoding and decoding for label images:

LabelColor = namedtuple("LabelColor", ["name","color"] )
labelColors = [
        LabelColor( name="Void", color=0 ),
        LabelColor( name="Liver", color=89 ),
        LabelColor( name="Gallbladder", color=124 ),
        LabelColor( name="Diaphragm", color=149 ),
        LabelColor( name="Fat", color=170 ),
        LabelColor( name="Ligament", color=188 ),
        LabelColor( name="ToolTip", color=203 ),
        LabelColor( name="ToolShaft", color=218 )
        ]

#labelColors = [
#        LabelColor( name="Liver", color=(111,71,62) ),
#        LabelColor( name="ToolShaft", color=(76,76,76) ),
#        LabelColor( name="ToolTip", color=(178,178,178) ),
#        LabelColor( name="Fat", color=(153,241,102) ),
#        LabelColor( name="Gallbladder", color=(153,241,171) ),
#        LabelColor( name="Diaphragm", color=(237,239,203) ),
#        LabelColor( name="Void", color=(0,0,0) )
#        ]


# Take a tensor with given colors and replace them with IDs:
def label2SingleChannel( colImage ):
    lbl = torch.LongTensor( colImage.shape[1:] )
    lbl.fill_( 255 )
    ligamentID, fatID = None, None
    for i in range( 0, len(labelColors) ):
        lc = labelColors[i]
        #mask = (colImage[0,:,:] == lc.color[0])*(colImage[1,:,:] == lc.color[1])*(colImage[2,:,:] == lc.color[2])
        mask = (colImage[0,:,:] == lc.color)
        #mask = colImage[:,:] == lc.color
        lbl[mask] = i
        if lc.name == "Fat":
            fatID = i
        if lc.name == "Ligament":
            ligamentID = i
    lbl[lbl==255] = 0
    # Replace "ligament" with "fat", as they aren't really distinguishable in a lot of cases:
    lbl[lbl==ligamentID] = fatID
    return lbl

def label2Color( lbl ):
    img = torch.FloatTensor( lbl.shape )
    img.fill_( 0 )
    img = img.repeat( (3,1,1) )
    lbl = lbl.squeeze()
    for i in range( 0, len(labelColors) ):
        lc = labelColors[i]
        mask = (lbl == i)
        img[:,mask] = torch.FloatTensor( [lc.color] ).repeat(3).view( 3, 1 )
    return img

#def oneHot2Label( lbl ):
#    inds = torch.argmax( lbl, dim=1 )
#    colImage = torch.zeros( (inds.shape[0], 3, inds.shape[1], inds.shape[2]) )
#    for im in range( 0, lbl.shape[0] ):
#        #col = labelColors[i].color
#        #print(torch.tensor(col))
#        for x in range( 0, lbl.shape[2] ):
#            for y in range( 0, lbl.shape[3] ):
#                colImage[im,:,x,y] = torch.tensor( labelColors[inds[im,x,y]].color )
#        #colImage[i = torch.where( inds==i, torch.tensor( col ), colImage )
#    
#    return colImage

