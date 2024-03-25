"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from utils import get_all_data_loaders, prepare_sub_folder, write_html, write_loss, get_config, write_2images, Timer, render_collate_fn
import argparse
from torch.autograd import Variable
from trainer import MUNIT_Trainer
import torch.backends.cudnn as cudnn
import torch
try:
	from itertools import izip as zip
except ImportError: # will be 3.x series
	pass
import os
import sys
import tensorboardX
import shutil
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

print('Remote server has been synchronized!')
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/simulation2surgery.yaml', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='/mnt/storage/home/lchen6/lchen6/data/Surgical/output/', help="output path")
# parser.add_argument("--resume", default=True, action="store_true")
parser.add_argument("--resume", action="store_true")

opts = parser.parse_args()

cudnn.benchmark = True

# Load experiment setting
config = get_config(opts.config)
max_iter = config['max_iter']
display_size = config['display_size']

# Setup model and data loader
train_loader_a, train_loader_b, test_loader_a, test_loader_b, num_scenes = get_all_data_loaders(config)

config['num_scenes'] = num_scenes
trainer = MUNIT_Trainer(config).cuda()

random.seed(1)
train_indices_a = []
train_indices_b = []
test_indices_a = []
test_indices_b = []
for i in range(display_size):
	train_indices_a.append(random.randrange(len(train_loader_a.dataset)))
	train_indices_b.append(random.randrange(len(train_loader_b.dataset)))
	test_indices_a.append(random.randrange(len(test_loader_a.dataset)))
	test_indices_b.append(random.randrange(len(test_loader_b.dataset)))
train_display_images_b = torch.stack([train_loader_b.dataset[i] for i in train_indices_b]).cuda()
test_display_images_b = torch.stack([test_loader_b.dataset[i] for i in test_indices_b]).cuda()


# Setup logger and output folders
model_name = os.path.splitext(os.path.basename(opts.config))[0]
train_writer = tensorboardX.SummaryWriter(os.path.join(opts.output_path + "/logs", model_name))
output_directory = os.path.join(opts.output_path + "/outputs", model_name)
checkpoint_directory, image_directory = prepare_sub_folder(output_directory)

# Backup copy of current settings and scripts:
shutil.copyfile(opts.config, os.path.join(output_directory, 'config.yaml')) # copy config file to output folder
runPath = os.path.dirname(os.path.realpath(__file__))
for f in os.listdir(runPath):
	if f.endswith(".py"):
		shutil.copyfile(os.path.join(runPath, f), os.path.join(output_directory, f)) # copy config file to output folder

# Start training
iterations = trainer.resume(checkpoint_directory, hyperparameters=config) if opts.resume else 0
config['use_view_loss'] = False
while True:
	for it, (batch_a, images_b) in enumerate(zip(train_loader_a, train_loader_b)):
		# if iterations == 10000:
		# 	config['use_view_loss'] = True
		with Timer("Elapsed time in update: %f"):

			#images_a = trainer.render_a(renderdata_a)
			images_b = images_b.cuda().detach()
			# Main training code
			#trainer.seg_update(images_a, labels_a, config)
			trainer.dis_update(batch_a, images_b, config)
			trainer.gen_update(batch_a, images_b, config)
			torch.cuda.synchronize()
		trainer.update_learning_rate()

		# Dump training stats in log file
		if (iterations + 1) % config['log_iter'] == 0:
			print("Iteration: %08d/%08d" % (iterations + 1, max_iter))
			write_loss(iterations, trainer, train_writer)

		# Write images
		if (iterations + 1) % config['image_save_iter'] == 0:
			with torch.no_grad():
				train_display_batch_a = render_collate_fn([train_loader_a.dataset[i] for i in train_indices_a])
				test_display_batch_a = render_collate_fn([test_loader_a.dataset[i] for i in test_indices_a])

				test_image_outputs = trainer.sample(test_display_batch_a, test_display_images_b)
				train_image_outputs = trainer.sample(train_display_batch_a, train_display_images_b)
			write_2images(test_image_outputs, display_size, image_directory, 'test_%08d' % (iterations + 1))
			write_2images(train_image_outputs, display_size, image_directory, 'train_%08d' % (iterations + 1))
			# HTML
			write_html(output_directory + "/index.html", iterations + 1, config['image_save_iter'], 'images')
			del test_image_outputs, train_image_outputs, train_display_batch_a, test_display_batch_a

		if (iterations + 1) % config['image_display_iter'] == 0:
			with torch.no_grad():

				train_display_batch_a = render_collate_fn([train_loader_a.dataset[i] for i in train_indices_a])
				test_display_batch_a = render_collate_fn([test_loader_a.dataset[i] for i in test_indices_a])

				test_image_outputs = trainer.sample(test_display_batch_a, test_display_images_b)
				train_image_outputs = trainer.sample(train_display_batch_a, train_display_images_b)
			write_2images(test_image_outputs, display_size, image_directory, 'test_current')
			write_2images(train_image_outputs, display_size, image_directory, 'train_current')
			del test_image_outputs, train_image_outputs, train_display_batch_a, test_display_batch_a

		# Save network weights
		if (iterations + 1) % config['snapshot_save_iter'] == 0:
			trainer.save(checkpoint_directory, iterations)

		iterations += 1
		if iterations >= max_iter:
			sys.exit('Finish training')
