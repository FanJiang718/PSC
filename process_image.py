#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

import fcn_zhang
import dataprocessing

import os, sys

from matplotlib import cm

default_cmap = cm.gist_heat


from PIL import Image, ImageDraw

import time

default_weightfile = "default-weights-textdetect.h5"





def apply_to_image(network,imgfile,overlap_factor=2.,tile_size=(224,224)):
	sx, sy = tile_size
	imgarray = dataprocessing.load_img(imgfile)
	w,h = imgarray.shape[:2]
	heatmap = np.zeros((w,h))
	normfactor = np.zeros((w,h))
	if(hasattr(overlap_factor,"__iter__")):
		overlap = overlap_factor
	else:
		overlap = (dataprocessing.compute_overlap(w,tile_size[0], overlap_factor),dataprocessing.compute_overlap(h,tile_size[1], overlap_factor))
	ol_x, ol_y = overlap
	n_total = len(range(0,w-sx,sx-ol_x))*len(range(0,h-sy,sy-ol_y))
	steps = n_total/20
	print 'Starting [                    ]',
	print '\b'*22,
	sys.stdout.flush()
	k=0
	before = time.time()
	for i in range(0,w-sx,sx-ol_x):
		for j in range(0,h-sy,sy-ol_y):
			k += 1
			tiled_x = np.copy(imgarray[i:i+sx,j:j+sy,:])
			net_input = np.expand_dims(tiled_x,0).transpose((0,3,1,2))
			out = network.predict(net_input)[0,0]
			heatmap[i:i+sx,j:j+sy] += out
			normfactor[i:i+sx,j:j+sy] += 1.
			if(k % steps ==0):
				print '\b.',
				sys.stdout.flush()
	after = time.time()
	print '\b]  Done!'
	print("Image was processed in %.2f seconds" % (after-before))
	normfactor = np.maximum(normfactor,1.)
	return heatmap/normfactor

def hmap_to_img(hmap,cmap=default_cmap):
	hmapimage = Image.fromarray(np.uint8(cmap(hmap)*255))
	return hmapimage


def process_image(network,imgfile,outpath='.',overlap_factor=1.5,tile_size=(224,224),cmap=default_cmap):
	filebase = imgfile.split('/')[-1].split('.')[0]
	base_img = Image.open(imgfile).convert('RGB')
	print("Processing image...")
	heatmap = apply_to_image(network,imgfile,overlap_factor,tile_size)
	print("Saving heatmap...")
	np.savez("%s/%s-heatmap.npz" % (outpath,filebase), heatmap)
	print("Computing and saving heatmap image...")
	heat_img = hmap_to_img(heatmap).convert('RGB')
	heat_img.save("%s/%s-heat.png" % (outpath,filebase))
	print("Processing and blending heatmap...")
	blended_img = Image.blend(base_img,heat_img,0.5)
	print("Saving all...")
	blended_img.save("%s/%s-blend.png" % (outpath,filebase))
	return heatmap



if __name__ == "__main__":
	print("\n")
	def usageexit():
		print("Usage: %s base_image_file [weights_file] [output_path] [overlap_factor]" % sys.argv[0])
		print("\n")
		sys.exit(1)

	if(len(sys.argv) < 2):
		usageexit()

	imgfile = sys.argv[1].strip()
	print("Processing image %s" % imgfile)

	if(len(sys.argv) < 3):
		if(not os.path.isfile(default_weightfile)):
			print("No weights file provided and default weights file %s not found" % default_weightfile)
			usageexit()
		else:
			print("Using default weights file %s" % default_weightfile)
			weightsfile = default_weightfile
	else:
		weightsfile = sys.argv[2].strip()
		print("Using custom weights file %s" % weightsfile)
	
	if(len(sys.argv) > 3):
		outpath = sys.argv[3].strip()
	else:
		outpath = "."
		
	if(len(sys.argv) > 4):
		overlap = float(sys.argv[4].strip())
	else:
		overlap = 2.
	print ("Using overlap factor of %f" % overlap)
	print("=====")
	print("Building network...")
	model = fcn_zhang.zhang_network(weights_path=None)
	print("Loading weights...")
	model.load_weights(weightsfile)
	process_image(model,imgfile,outpath=outpath,overlap_factor=overlap)
	print("Files saved in %s" % outpath)
	print("=====")
	print("Done")

