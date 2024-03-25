import sys
sys.path.append('/mnt/storage/home/lchen6/anaconda3/envs/CIIGAN/lib/python3.7/site-packages/')
import torch

class Renderer():

	def __init__( self, imW, imH, texW=0, texH=0 ):

		self.imW = imW
		self.imH = imH
		self.texW = texW
		self.texH = texH

		self.indSrc = torch.arange(imH*imW).view(imH,imW).transpose(1,0).reshape(-1).cuda()

	'''def corrsToTorch(self,correspondences):
		indTex, indIm, weights = [], [], []
		# Fill index lists:
		validCorrespondences = 0
		for c in correspondences:
			texX, texY = c.srcCoord.x, c.srcCoord.y
			if texX < 0 or texY < 0:
				weights += [0,0,0,0]
			else:
				iTexX,iTexY = int(texX), int(texY)

				# Find the (continuous, linear) index for the pixel in the image and save it:
				targetX = int(c.tgtCoord.x)
				targetY = int(c.tgtCoord.y)
				targetCoord = self.continuousIndex( self.imW, targetX, targetY )
				indIm.append(targetCoord)

				# The texture coordinates could be floating point, so find each of the neighbouring
				# four pixels:
				srcCoord11 = self.continuousIndex( self.texW, iTexX, iTexY )
				srcCoord21 = self.continuousIndex( self.texW, iTexX+1, iTexY )
				srcCoord12 = self.continuousIndex( self.texW, iTexX, iTexY+1 )
				srcCoord22 = self.continuousIndex( self.texW, iTexX+1, iTexY+1 )
				indTex += [srcCoord11, srcCoord21, srcCoord12, srcCoord22]

				# Find out how each of these four pixels should be weighted, and save this as well:
				x1y1, x2y1, x1y2, x2y2 = self.bilinearCoeff( c.srcCoord.x, c.srcCoord.y )
				weights += [x1y1,x2y1,x1y2,x2y2]
				validCorrespondences += 1

		# Turn weight and index lists into tensors.
		indTex = torch.LongTensor( indTex ).view( validCorrespondences, 4 ).permute( 1,0 )
		indIm = torch.LongTensor( indIm )
		weights = torch.FloatTensor( weights ).view( self.imW, self.imH, 4 ).permute( 2,1,0 )

		# Repeat the tensors for the three color channels:
		indTex
		indIm
		return indTex, indIm, weights

	def continuousIndex( self, w, x, y ):
		return y*w + x

	def bilinearCoeff( self, x, y ):
		dxLow = (x-int(x))
		dxHigh = (int(x)+1-x)
		dyLow = (y-int(y))
		dyHigh = (int(y)+1-y)
		return dxHigh*dyHigh, dxLow*dyHigh, dxHigh*dyLow, dxLow*dyLow'''

	def renderTextureView( self, tex, indTex, indIm, weights):
		# Gather the values required from the texture, then distribute them to the correct pixels in the image.
		# Repeat this four times, once for each part of the bilinear interpolation.
		# Multiply each part with the weights for these pixels
		scattered_weighted = []
		for i in range(4):
			gathered = torch.gather( tex, 1, indTex[:,i,:] )
			scattered = torch.zeros( (3, self.imH*self.imW) ).cuda().scatter( 1, indIm, gathered )
			scattered_weighted.append(scattered.view(3,self.imH,self.imW)*weights[i,:])
		# Then sum up the parts to create the final image:
		return sum(scattered_weighted)

	def render(self,tex,indTexList,indImList,weightList):
		return sum([self.renderTextureView(tex,indTex.repeat(3,1,1),indIm.repeat(3,1),weights) for indTex,indIm,weights in zip(indTexList,indImList,weightList)])

	def luminosity(self,depth,a=.000035, b=.0004, c=.4):
		depth = torch.clamp(depth, max=100)
		# a=.000035, b=.0004, c=.4 # without clamping
		# a=.00005,b=.001,c=.3 # without clamping
		attenuation = a*(depth**2) + b*depth + c
		luminosity = 1 / attenuation
		return luminosity

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