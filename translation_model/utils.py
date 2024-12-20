"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
#from torch.utils.serialization import load_lua
from torch.utils.data import DataLoader
from networks import Vgg16
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import transforms
from data import ImageFilelist, ImageFolder, RenderFolder
import torch
import torch.nn as nn
import os
import math
import torchvision.utils as vutils
import yaml
import numpy as np
import torch.nn.init as init
import time
# Methods
# get_all_data_loaders      : primary data loader interface (load trainA, testA, trainB, testB)
# get_data_loader_list      : list-based data loader
# get_data_loader_folder    : folder-based data loader
# get_config                : load yaml file
# eformat                   :
# write_2images             : save output image
# prepare_sub_folder        : create checkpoints and images folders for saving outputs
# write_one_row_html        : write one row of the html file for output images
# write_html                : create the html file.
# write_loss
# slerp
# get_slerp_interp
# get_model_list
# load_vgg16
# load_inception
# vgg_preprocess
# get_scheduler
# weights_init

def get_all_data_loaders(conf):
    batch_size = conf['batch_size']
    num_workers = conf['num_workers']
    height = conf['crop_image_height']
    width = conf['crop_image_width']

    if 'data_root' in conf:
        # /raid/compass/backup/output_long/data
        train_fake_path = '/mnt/storage/home/lchen6/lchen6/data/Surgical/simulate_images'
        train_real_path = '/mnt/storage/home/lchen6/lchen6/data/Surgical/real_images/'
        # test_fake_path = train_fake_path + '_sequences'
        test_fake_path = train_fake_path
        test_real_path = train_real_path

        scenes = sorted(os.listdir(train_fake_path))
        scene_to_index = {scene: i for i,scene in enumerate(scenes)}
        num_scenes = len(scenes)

        aug = {}
        aug["new_size_min"] = conf["new_size_min_a"]
        aug["new_size_max"] = conf["new_size_max_a"]
        aug["output_size"] = (width, height)
        aug["circle_mask"] = False
        aug["rotate"] = False
        aug["contrast"] = False
        train_loader_a = get_renderdata_loader(train_fake_path, scene_to_index, batch_size, True, num_workers, augmentation=aug )
        test_loader_a = get_renderdata_loader(test_fake_path, scene_to_index, batch_size, False, num_workers, augmentation=aug )

        aug["new_size_min"] = conf["new_size_min_b"]
        aug["new_size_max"] = conf["new_size_max_b"]
        aug["circle_mask"] = False
        aug["contrast"] = False
        train_loader_b = get_image_loader(train_real_path, batch_size, True, num_workers, load_labels=False, augmentation=aug )
        test_loader_b = get_image_loader(test_real_path, batch_size, False, num_workers, load_labels=False, augmentation=aug )
        print("train_loader_a", len(train_loader_a))
        print("train_loader_b", len(train_loader_b))
        print("test_loader_a", len(test_loader_a))
        print("test_loader_b", len(test_loader_b))
    else:
        raise IOError("Please provide a 'data_root' folder in the config file!")
    return train_loader_a, train_loader_b, test_loader_a, test_loader_b, num_scenes

def get_image_loader(input_folder, batch_size, train, num_workers=4, load_labels=False, load_paths=False, augmentation={}):
    dataset = ImageFolder(input_folder, return_labels=load_labels, return_paths=load_paths, augmentation=augmentation)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=train, drop_last=True, num_workers=num_workers)
    return loader

def get_renderdata_loader(input_folder, scene_to_index, batch_size, train, num_workers=4, translation_mode=False, load_paths=False, augmentation={}):
    dataset = RenderFolder(input_folder, scene_to_index, translation_mode)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=train, drop_last=True, num_workers=num_workers, collate_fn=render_collate_fn)
    return loader

def render_collate_fn(batch):
    renderdata = [i[0] for i in batch]
    renderdata_2 = [i[1] for i in batch]
    scene_idx = [i[2] for i in batch]
    return renderdata, renderdata_2, scene_idx

#def render_collate_fn(batch):
#    num_objects = len(batch[0])
#    return ([el[j] for el in batch] for j in range(num_objects))

def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)

def eformat(f, prec):
    s = "%.*e"%(prec, f)
    mantissa, exp = s.split('e')
    # add 1 to digits as 1 is taken by sign +/-
    return "%se%d"%(mantissa, int(exp))


def __write_images(image_outputs, display_image_num, file_name):
    image_outputs = [images.expand(-1, 3, -1, -1) for images in image_outputs] # expand gray-scale images to 3 channels
    image_tensor = torch.cat([images[:display_image_num] for images in image_outputs], 0)
    image_grid = vutils.make_grid(image_tensor.data, nrow=display_image_num, padding=0)
    vutils.save_image(image_grid, file_name, nrow=1)


def write_2images(image_outputs, display_image_num, image_directory, postfix):
    n = len(image_outputs)
    __write_images(image_outputs[0:6], display_image_num, '%s/gen_a2b_%s.png' % (image_directory, postfix))
    __write_images(image_outputs[6:n], display_image_num, '%s/gen_b2a_%s.png' % (image_directory, postfix))


def prepare_sub_folder(output_directory):
    image_directory = os.path.join(output_directory, 'images')
    if not os.path.exists(image_directory):
        print("Creating directory: {}".format(image_directory))
        os.makedirs(image_directory)
    checkpoint_directory = os.path.join(output_directory, 'checkpoints')
    if not os.path.exists(checkpoint_directory):
        print("Creating directory: {}".format(checkpoint_directory))
        os.makedirs(checkpoint_directory)
    return checkpoint_directory, image_directory


def write_one_row_html(html_file, iterations, img_filename, all_size):
    html_file.write("<h3>iteration [%d] (%s)</h3>" % (iterations,img_filename.split('/')[-1]))
    html_file.write("""
        <p><a href="%s">
          <img src="%s" style="width:%dpx">
        </a><br>
        <p>
        """ % (img_filename, img_filename, all_size))
    return


def write_html(filename, iterations, image_save_iterations, image_directory, all_size=1536):
    html_file = open(filename, "w")
    html_file.write('''
    <!DOCTYPE html>
    <html>
    <head>
      <title>Experiment name = %s</title>
      <meta http-equiv="refresh" content="30">
    </head>
    <body>
    ''' % os.path.basename(filename))
    html_file.write("<h3>current</h3>")
    write_one_row_html(html_file, iterations, '%s/gen_a2b_train_current.png' % (image_directory), all_size)
    write_one_row_html(html_file, iterations, '%s/gen_b2a_train_current.png' % (image_directory), all_size)
    for j in range(iterations, image_save_iterations-1, -1):
        if j % image_save_iterations == 0:
            write_one_row_html(html_file, j, '%s/gen_a2b_test_%08d.png' % (image_directory, j), all_size)
            write_one_row_html(html_file, j, '%s/gen_b2a_test_%08d.png' % (image_directory, j), all_size)
            write_one_row_html(html_file, j, '%s/gen_a2b_train_%08d.png' % (image_directory, j), all_size)
            write_one_row_html(html_file, j, '%s/gen_b2a_train_%08d.png' % (image_directory, j), all_size)
    html_file.write("</body></html>")
    html_file.close()


def write_loss(iterations, trainer, train_writer):
    members = [attr for attr in dir(trainer) \
               if not callable(getattr(trainer, attr)) and not attr.startswith("__") and ('loss' in attr or 'grad' in attr or 'nwd' in attr)]
    for m in members:
        train_writer.add_scalar(m, getattr(trainer, m), iterations + 1)


def slerp(val, low, high):
    """
    original: Animating Rotation with Quaternion Curves, Ken Shoemake
    https://arxiv.org/abs/1609.04468
    Code: https://github.com/soumith/dcgan.torch/issues/14, Tom White
    """
    omega = np.arccos(np.dot(low / np.linalg.norm(low), high / np.linalg.norm(high)))
    so = np.sin(omega)
    return np.sin((1.0 - val) * omega) / so * low + np.sin(val * omega) / so * high


def get_slerp_interp(nb_latents, nb_interp, z_dim):
    """
    modified from: PyTorch inference for "Progressive Growing of GANs" with CelebA snapshot
    https://github.com/ptrblck/prog_gans_pytorch_inference
    """

    latent_interps = np.empty(shape=(0, z_dim), dtype=np.float32)
    for _ in range(nb_latents):
        low = np.random.randn(z_dim)
        high = np.random.randn(z_dim)  # low + np.random.randn(512) * 0.7
        interp_vals = np.linspace(0, 1, num=nb_interp)
        latent_interp = np.array([slerp(v, low, high) for v in interp_vals],
                                 dtype=np.float32)
        latent_interps = np.vstack((latent_interps, latent_interp))

    return latent_interps[:, :, np.newaxis, np.newaxis]


# Get model list for resume
def get_model_list(dirname, key):
    if os.path.exists(dirname) is False:
        return None
    gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
                  os.path.isfile(os.path.join(dirname, f)) and key in f and ".pt" in f]
    if gen_models is None:
        return None

    gen_models.sort()
    last_model_name = gen_models[-1]
    return last_model_name


def get_scheduler(optimizer, hyperparameters, iterations=-1):
    if 'lr_policy' not in hyperparameters or hyperparameters['lr_policy'] == 'constant':
        scheduler = None # constant scheduler
    elif hyperparameters['lr_policy'] == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=hyperparameters['step_size'],
                                        gamma=hyperparameters['gamma'], last_epoch=iterations)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', hyperparameters['lr_policy'])
    return scheduler


def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            # print m.__class__.__name__
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    return init_fun


class Timer:
    def __init__(self, msg):
        self.msg = msg
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_value, exc_tb):
        print(self.msg % (time.time() - self.start_time))

def acos(x, epsilon=1e-7):
    return torch.acos(torch.clamp(x, -1 + epsilon, 1 - epsilon))