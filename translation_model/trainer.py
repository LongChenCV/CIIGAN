"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from networks import AdaINGen, MsImageDis, VAEGen, StylelessGen, ResBlockSegmentation
from utils import weights_init, get_model_list, get_scheduler
from utils import __write_images as writeImage
from torch.autograd import Variable
from pytorch_msssim import msssim, ssim
import torch
import torch.nn as nn
import os
import torchvision
import random
import rendering
import utils

class MUNIT_Trainer(nn.Module):
	def __init__(self, hyperparameters):
		super(MUNIT_Trainer, self).__init__()
		lr = hyperparameters['lr']

		self.gen_a = StylelessGen(hyperparameters['input_dim_a'], hyperparameters['gen'])  # auto-encoder for domain a
		self.gen_b = StylelessGen(hyperparameters['input_dim_b'], hyperparameters['gen'])  # auto-encoder for domain b
		self.dis_a = MsImageDis(hyperparameters['input_dim_a'], hyperparameters['dis'])  # discriminator for domain a
		self.dis_b = MsImageDis(hyperparameters['input_dim_b'], hyperparameters['dis'])  # discriminator for domain b

		self.instancenorm = nn.InstanceNorm2d(512, affine=False)
		self.style_dim = hyperparameters['gen']['style_dim']
		self.num_classes = hyperparameters['gen']['num_classes']

		imW, imH = hyperparameters['crop_image_width'], hyperparameters['crop_image_height']
		texPatchSize = hyperparameters['tex_patch_size']
		self.renderer = rendering.Renderer(imW, imH, texPatchSize)

		self.tex_ref = self.renderer.init_texture()
		self.tex = torch.stack([self.renderer.init_texture() for _ in range(hyperparameters['num_scenes'])])
		self.tex.requires_grad = True

		pi = torch.acos(torch.zeros(1)).item() * 2
		cos_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
		self.cos_dis = lambda x1,x2: 1 - cos_sim(x1,x2)
		self.ang_dis = lambda x1,x2: (2*utils.acos(cos_sim(x1,x2))) / pi

		# fix the noise used in sampling
		display_size = int(hyperparameters['display_size'])

		# Setup the optimizers
		beta1 = hyperparameters['beta1']
		beta2 = hyperparameters['beta2']
		dis_params = list(self.dis_a.parameters()) + list(self.dis_b.parameters())
		gen_params = list(self.gen_a.parameters()) + list(self.gen_b.parameters())

		self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
										lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
		self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
										lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
		self.tex_opt = torch.optim.Adam([self.tex],
										lr=lr*10, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
		self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters)
		self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)
		self.tex_scheduler = get_scheduler(self.tex_opt, hyperparameters)

		# Network weight initialization
		self.apply(weights_init(hyperparameters['init']))
		self.dis_a.apply(weights_init('gaussian'))
		self.dis_b.apply(weights_init('gaussian'))

		self.loss_view = 0

	def recon_criterion(self, input, target):
		return torch.mean(torch.abs(input - target))

	def forward(self, x_a, x_b):
		self.eval()
		#s_a = Variable(self.s_a)
		s_b = Variable(self.s_b)
		c_a = self.gen_a.encode(x_a)
		c_b, s_b_fake = self.gen_b.encode(x_b)
		x_ba = self.gen_a.decode(c_b)
		x_ab = self.gen_b.decode(c_a, s_b)
		self.train()
		return x_ab, x_ba

	def render_a(self, renderdata, scene_idx):

		texBatch = torch.tanh(self.tex[scene_idx])
		
		depth = torch.stack([c[3] for c in renderdata])
		diffusion = torch.stack([c[4] for c in renderdata])
		corrBatch = [c[:3] for c in renderdata]

		alpha = .9
		luminosity = self.renderer.luminosity(depth)

		reference = self.renderer.render_batch(self.tex_ref, corrBatch)
		reference = reference + 1
		reference = reference * (diffusion*alpha + (1-alpha))
		reference = reference * luminosity
		reference = reference - 1
		textured = self.renderer.render_batch(texBatch, corrBatch)
		textured = torch.cat((textured, reference),dim=1)
		return textured

	def get_warped_view(self, renderdata, renderdata_2, scene_idx):
		with torch.no_grad():
			# synthesize 2nd view
			x = self.render_a(renderdata_2,scene_idx)
			# Add CII or LBP features
			x_a_lbp = torch.stack([c[7] for c in renderdata_2])
			x_a_lbp = x_a_lbp.to(x.device)
			x = torch.cat((x, x_a_lbp), dim=1)

			c = self.gen_a.encode(x)
			x = self.gen_b.decode(c)
			# projection matrix of 1st view
			projectionMatrix = torch.stack([c[5] for c in renderdata])
			# 3D locations of both views
			points3D = torch.stack([c[6] for c in renderdata])
			points3D_2 = torch.stack([c[6] for c in renderdata_2])
			# warp
			x_w, depth_w = self.renderer.warp(projectionMatrix,points3D_2,x)
			depth = self.renderer.get_z(projectionMatrix,points3D)
			x_w = self.renderer.remove_occlusions(x_w,depth_w,depth)
		return x_w.detach(), x.detach()

	def view_loss(self,img,warped):

		warped = warped.detach()
		
		mask = (warped != 0)
		mask,_ = mask.min(dim=1)
		num_warped_pixels = mask.sum()
		if num_warped_pixels == 0:
			num_warped_pixels = 1

		img, warped = (img+1)/2, (warped+1)/2

		error = self.ang_dis(img,warped)
		error = torch.where(mask, error, torch.cuda.FloatTensor([0]))
		return error.sum() / num_warped_pixels

	def gen_update(self, x_a, x_b, hyperparameters, useLabelLoss=False):
		self.gen_opt.zero_grad()
		self.tex_opt.zero_grad()
		renderdata, renderdata_2, scene_idx = x_a
		x_a = self.render_a(renderdata,scene_idx)
		# Add CII or LBP features
		x_a_lbp= torch.stack([c[7] for c in renderdata])
		x_a_lbp = x_a_lbp.to(x_a.device)
		x_a = torch.cat((x_a, x_a_lbp), dim=1)

		# encode
		c_a = self.gen_a.encode(x_a)
		c_b = self.gen_b.encode(x_b)
		# decode (cross domain)
		x_ba = self.gen_a.decode(c_b)
		x_ab = self.gen_b.decode(c_a)

		# decode (within domain)
		x_a_recon = self.gen_a.decode(c_a)
		x_b_recon = self.gen_b.decode(c_b)

		# Structural similarity:
		x_a_ref = x_a[:, 3:]
		x_ba_ref = x_ba[:, 3:]
		x_a_brightness = torch.mean( x_a_ref, dim=1, keepdim=True )
		x_b_brightness = torch.mean( x_b, dim=1, keepdim=True )
		x_ab_brightness = torch.mean( x_ab, dim=1, keepdim=True )
		x_ba_brightness = torch.mean( x_ba_ref, dim=1, keepdim=True )
		loss_msssim_ab = -msssim(x_a_brightness, x_ab_brightness, normalize=True)
		loss_msssim_ba = -msssim(x_b_brightness, x_ba_brightness, normalize=True)
		#loss_msssim_ab = -msssim(x_a_ref, x_ab, normalize=True)
		#loss_msssim_ba = -msssim(x_b, x_ba_ref, normalize=True)

		# encode again
		c_b_recon = self.gen_a.encode(x_ba)
		c_a_recon = self.gen_b.encode(x_ab)
		# decode again (if needed)
		x_aba = self.gen_a.decode(c_a_recon) if hyperparameters['recon_x_cyc_w'] > 0 else None
		x_bab = self.gen_b.decode(c_b_recon) if hyperparameters['recon_x_cyc_w'] > 0 else None

		# reconstruction loss
		loss_gen_recon_x_a = self.recon_criterion(x_a_recon, x_a)
		loss_gen_recon_x_b = self.recon_criterion(x_b_recon, x_b)
		loss_gen_recon_c_a = self.recon_criterion(c_a_recon, c_a)
		loss_gen_recon_c_b = self.recon_criterion(c_b_recon, c_b)
		loss_gen_cycrecon_x_a = self.recon_criterion(x_aba, x_a) if hyperparameters['recon_x_cyc_w'] > 0 else 0
		loss_gen_cycrecon_x_b = self.recon_criterion(x_bab, x_b) if hyperparameters['recon_x_cyc_w'] > 0 else 0
		# GAN loss
		loss_gen_adv_a = self.dis_a.calc_gen_loss(x_ba)
		loss_gen_adv_b = self.dis_b.calc_gen_loss(x_ab)
		# total loss
		loss_gen_total = hyperparameters['gan_w'] * loss_gen_adv_a + \
						  hyperparameters['gan_w'] * loss_gen_adv_b + \
						  hyperparameters['recon_x_w'] * loss_gen_recon_x_a + \
						  hyperparameters['recon_c_w'] * loss_gen_recon_c_a + \
						  hyperparameters['recon_x_w'] * loss_gen_recon_x_b + \
						  hyperparameters['recon_c_w'] * loss_gen_recon_c_b + \
						  hyperparameters['recon_x_cyc_w'] * loss_gen_cycrecon_x_a + \
						  hyperparameters['recon_x_cyc_w'] * loss_gen_cycrecon_x_b + \
						  hyperparameters['ms_ssim_a_w']*loss_msssim_ab + \
						  hyperparameters['ms_ssim_b_w']*loss_msssim_ba

		# view-consistency loss
		if hyperparameters['use_view_loss']:
			x_w, x_ab_2 = self.get_warped_view(renderdata, renderdata_2, scene_idx)
			loss_view = self.view_loss(x_ab,x_w)
			loss_gen_total += hyperparameters['view_w']*loss_view
			self.loss_view = loss_view.item()

		loss_gen_total.backward()

		self.gen_opt.step()
		self.tex_opt.step()

		self.loss_gen_adv_a = loss_gen_adv_a.item()
		self.loss_gen_adv_b = loss_gen_adv_a.item()
		self.loss_gen_recon_x_a = loss_gen_recon_x_a.item()
		self.loss_gen_recon_c_a = loss_gen_recon_c_a.item()
		self.loss_gen_recon_x_b = loss_gen_recon_x_b.item()
		self.loss_gen_recon_c_b = loss_gen_recon_c_b.item()
		self.loss_gen_cycrecon_x_a = loss_gen_cycrecon_x_a.item()
		self.loss_gen_cycrecon_x_b = loss_gen_cycrecon_x_b.item()
		self.loss_msssim_ab = loss_msssim_ab.item()
		self.loss_msssim_ba = loss_msssim_ba.item()

		self.loss_gen_total = loss_gen_total.item()

	def sample(self, x_a, x_b):

		self.eval()

		renderdata, _, scene_idx = x_a
		x_a = self.render_a(renderdata,scene_idx)
		# Add CII or LBP features
		x_a_lbp = torch.stack([c[7] for c in renderdata])
		x_a_lbp = x_a_lbp.to(x_a.device)

		x_a = torch.cat((x_a, x_a_lbp), dim=1)

		x_a_recon, x_b_recon, x_ba, x_bab, x_ab, x_aba = [], [], [], [], [], []
		for i in range(x_a.size(0)):
			# get individual images from list:
			x_a_ = x_a[i].unsqueeze(0)
			x_b_ = x_b[i].unsqueeze(0)

			# a to b:
			c_a = self.gen_a.encode(x_a_)
			x_a_recon_ = self.gen_a.decode(c_a)     # Reconstruct in same domain
			c_b = self.gen_b.encode(x_b_)
			x_ab_ = self.gen_b.decode(c_a)     # translate
			c_ab = self.gen_b.encode(x_ab_) # re-encode
			x_aba_ = self.gen_a.decode(c_ab) # translate back

			x_a_recon.append(x_a_recon_)
			x_ab.append(x_ab_)
			x_aba.append(x_aba_)

			# b to a:
			x_ba_ = self.gen_a.decode(c_b)      # translate
			c_ba = self.gen_a.encode(x_ba_)   # re-encode
			x_b_recon_ = self.gen_b.decode(c_b)      # Reconstruct in same domain
			x_bab_ = self.gen_b.decode(c_ba)    # translate back

			x_b_recon.append(x_b_recon_)
			x_ba.append(x_ba_)
			x_bab.append(x_bab_)

		x_a = (x_a+1)/2
		x_b = (x_b+1)/2
		x_a_recon = (torch.cat(x_a_recon)+1)/2
		x_b_recon = (torch.cat(x_b_recon)+1)/2
		x_ba = (torch.cat(x_ba)+1)/2
		x_ab = (torch.cat(x_ab)+1)/2
		x_bab = (torch.cat(x_bab)+1)/2
		x_aba = (torch.cat(x_aba)+1)/2
		self.train()
		return x_a[:, 3:6], x_a[:, :3], x_a_recon[:, 3:6], x_ab, x_aba[:, 3:6], x_aba[:, :3], x_b, x_b_recon, x_ba[:, 3:6], x_bab

	def dis_update(self, x_a, x_b, hyperparameters):
		self.dis_opt.zero_grad()
		renderdata, _, scene_idx = x_a
		x_a = self.render_a(renderdata, scene_idx)

		# Add CII or LBP features
		x_a_lbp= torch.stack([c[7] for c in renderdata])
		x_a_lbp = x_a_lbp.to(x_a.device)
		x_a = torch.cat((x_a, x_a_lbp), dim=1)

		# encode
		c_a = self.gen_a.encode(x_a)
		c_b = self.gen_b.encode(x_b)
		# decode (cross domain)
		x_ba = self.gen_a.decode(c_b)
		x_ab = self.gen_b.decode(c_a)
		# D loss
		self.loss_dis_a = self.dis_a.calc_dis_loss(x_ba.detach(), x_a.detach())
		self.loss_dis_b = self.dis_b.calc_dis_loss(x_ab.detach(), x_b)
		self.loss_dis_total = hyperparameters['gan_w'] * self.loss_dis_a + hyperparameters['gan_w'] * self.loss_dis_b
		self.loss_dis_total.backward()
		self.dis_opt.step()

	def update_learning_rate(self):
		if self.dis_scheduler is not None:
			self.dis_scheduler.step()
		if self.gen_scheduler is not None:
			self.gen_scheduler.step()
		if self.tex_scheduler is not None:
			self.tex_scheduler.step()

	def resume(self, checkpoint_dir, hyperparameters):
		# Load generators
		last_model_name = get_model_list(checkpoint_dir, "gen")
		state_dict = torch.load(last_model_name)
		self.gen_a.load_state_dict(state_dict['a'])
		self.gen_b.load_state_dict(state_dict['b'])
		iterations = int(last_model_name[-11:-3])
		# Load discriminators
		last_model_name = get_model_list(checkpoint_dir, "dis")
		state_dict = torch.load(last_model_name)
		self.dis_a.load_state_dict(state_dict['a'])
		self.dis_b.load_state_dict(state_dict['b'])
		# Load optimizers
		state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
		self.dis_opt.load_state_dict(state_dict['dis'])
		self.gen_opt.load_state_dict(state_dict['gen'])
		# Reinitilize schedulers
		self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters, iterations)
		self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters, iterations)
		print('Resume from iteration %d' % iterations)
		return iterations

	def save(self, snapshot_dir, iterations):
		# Save generators, discriminators, and optimizers
		gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations + 1))
		dis_name = os.path.join(snapshot_dir, 'dis_%08d.pt' % (iterations + 1))
		opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
		torch.save({'a': self.gen_a.state_dict(), 'b': self.gen_b.state_dict(), 'tex': self.tex.detach(), 'tex_ref': self.tex_ref.detach()}, gen_name)
		torch.save({'a': self.dis_a.state_dict(), 'b': self.dis_b.state_dict()}, dis_name)
		torch.save({'gen': self.gen_opt.state_dict(), 'dis': self.dis_opt.state_dict()}, opt_name)
