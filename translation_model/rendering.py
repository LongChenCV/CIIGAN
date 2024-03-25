import torch
from torchvision import transforms
import os
from PIL import Image

class Renderer():

	def __init__(self, imW, imH, texPatchSize):

		self.imW = imW
		self.imH = imH
		self.texPatchSize = texPatchSize

		self.indSrc = torch.arange(imH*imW).view(imH,imW).transpose(1,0).reshape(-1).cuda()

	def render_texture_view(self, tex, indTex, indIm, weights):
		# Gather the values required from the texture, then distribute them to the correct pixels in the image.
		# Repeat this four times, once for each part of the bilinear interpolation.
		# Multiply each part with the weights for these pixels
		numChannels = tex.shape[0]
		indTex, indIm = indTex.repeat(numChannels,1,1), indIm.repeat(numChannels,1)
		scattered_weighted = []
		for i in range(4):
			gathered = torch.gather( tex, 1, indTex[:,i,:] )
			empty = torch.zeros((numChannels,self.imH*self.imW)).cuda()
			scattered = empty.scatter(1,indIm,gathered).view(numChannels,self.imH,self.imW)
			scattered = scattered * weights[i]
			scattered_weighted.append(scattered)

		# Then sum up the parts to create the final image:
		return sum(scattered_weighted)

	def render(self, tex, indTexList, indImList, weightList):
		return sum([self.render_texture_view(tex,indTex,indIm,weights)
					for indTex,indIm,weights
					in zip(indTexList,indImList,weightList)
					])

	def render_batch(self, texBatch, corrBatch):
		
		batch = []

		if texBatch.dim() == 3:
			for tex,(indTex,indIm,weights) in zip(texBatch,corrBatch):
				img = self.render(tex,indTex,indIm,weights)
				batch.append(img)
		elif texBatch.dim() == 2:
			tex = texBatch
			for indTex,indIm,weights in corrBatch:
				img = self.render(tex,indTex,indIm,weights)
				batch.append(img)

		return torch.stack(batch)

	def luminosity(self,depth,a=.000035, b=.0004, c=.4):
		depth = torch.clamp(depth, max=100)
		attenuation = a*(depth**2) + b*depth + c
		return 1 / attenuation
		
	def warp(self,projection_matrix,points3D,img):
		# image shape
		batch_size = img.shape[0]
		num_channels = img.shape[1]
		num_pixels = self.imH*self.imW
		# flatten image to shape (batch_size,num_channels,-1)
		img = img.view(batch_size,num_channels,num_pixels)
		# project 3D coordinates to 2D pixel coordinates
		indTgt = torch.squeeze(torch.matmul(projection_matrix,points3D),dim=-1)
		indTgt, z = torch.split(indTgt,[2,1],dim=-1)
		indTgt = torch.round(indTgt/z).type(torch.cuda.LongTensor)
		# overwrite pixels out of scope
		pX, pY = torch.split(indTgt,1,dim=-1)
		out_of_scope = (pX >= self.imW) | (pX < 0) | (pY >= self.imH) | (pY < 0)
		indTgt = torch.where(out_of_scope,torch.cuda.LongTensor([self.imW,self.imH-1]),indTgt)
		# convert to 1D coordinates (scatter and gather can only deal with 1D indices)
		pX, pY = torch.split(indTgt,1,dim=-1)
		indTgt = torch.squeeze(pX + pY*self.imW, dim=-1)
		# concatenate rgb with z value
		z = z.view(batch_size,self.imW,self.imH).transpose(2,1).reshape(batch_size,1,-1)
		img = torch.cat((img,z),dim=1)
		num_channels = img.shape[1]
		### map RGB values into target image ###
		indTgt = torch.unsqueeze(indTgt,dim=1).repeat(1,num_channels,1)
		indSrc = self.indSrc.repeat(batch_size,num_channels,1)
		# gather RGBD values form source image
		gathered = torch.gather(img,-1,indSrc)
		# scatter into target image
		# if multiple src values are mapped to the same target location, the mean is used
		# (inspired by https://github.com/rusty1s/pytorch_scatter/tree/1.4.0)
		empty_img = torch.zeros((batch_size,num_channels,num_pixels+1)).cuda()
		scattered = empty_img.scatter_add(-1,indTgt,gathered)
		count = empty_img.scatter_add(-1,indTgt,torch.ones_like(gathered))
		count = torch.where(count==0,torch.cuda.FloatTensor([1]),count)
		scattered = scattered / count
		# remove pixels out of range and reshape
		scattered = scattered[:,:,:-1]
		warped = scattered.view(batch_size,num_channels,self.imH,self.imW)
		warped, z = torch.split(warped,[3,1],dim=1)
		return warped, z

	def get_z(self,projection_matrix,points3D):
		ind = torch.squeeze(torch.matmul(projection_matrix,points3D),dim=-1)
		z = ind[:,:,-1]
		z = z.view(points3D.shape[0],1,self.imW,self.imH).transpose(3,2)
		return z

	def remove_occlusions(self,img,z,z_tgt,epsilon=1):

		if z_tgt.min() < 0:
			print('WARNING: NEGATIVE DEPTH DETECTED. This will cause the occlusion detection to fail.')

		occlusion = ((z-z_tgt) > epsilon) & (z != 0)
		img = torch.where(occlusion,torch.cuda.FloatTensor([0]),img)
		return img

	def init_texture(self, numObjects=5, numTexPatches=6):

		texW = self.texPatchSize*numObjects + 1
		texH = self.texPatchSize*numTexPatches + 1

		transform = transforms.Compose([
			transforms.Resize(self.texPatchSize),
			transforms.RandomCrop(self.texPatchSize),
			transforms.ToTensor()
		])
		tex_tmp = [
			transform(Image.open(os.path.join('texture_patches/',f)).convert('RGB'))
			for f
			in sorted(os.listdir('texture_patches'))
			if '.png' in f and '_' in f
		]
		tex_tmp = torch.cat(tex_tmp, dim=2).repeat((1,numTexPatches,1))
		tex = torch.zeros(3,texH,texW)
		tex[:,:-1,:-1] = tex_tmp
		tex = tex.view(3,-1).cuda()
		tex = (tex - .5) / .5
		return tex