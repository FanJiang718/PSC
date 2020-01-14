#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


import sys
import os
import io
import chardet
import codecs
import datetime

import cv2

from PIL import Image, ImageDraw

vgg_rgboffsets = [103.939, 116.779, 123.68]


image_fs = "rrc_t4/img/img_%d.jpg"
gt_fs = "rrc_t4/gt/gt_img_%d.txt"

def detect_encoding(fname):
	bytes = min(32, os.path.getsize(fname))
	raw = open(fname, 'rb').read(bytes)
	if raw.startswith(codecs.BOM_UTF8):
		encoding = 'utf-8-sig'
	else:
		result = chardet.detect(raw)
		encoding = result['encoding']
	return encoding



def process_groundtruth(gtfile,w=1280,h=720):
	image_match = Image.new('L',(w,h))
	image_dontcare = Image.new('L',(w,h))
	encoding = detect_encoding(gtfile)
	with io.open(gtfile,'r',encoding=encoding) as fd:
		for bbox_line in fd:
			fields = bbox_line.strip().split(',',8)
			#xvals = [int(x) for x in fields[:8:2]]
			#yvals = [int(y) for y in fields[1:8:2]]
			xycoords = [int(xy) for xy in fields[:-1]]
			label = fields[-1]
			draw = ImageDraw.Draw(image_dontcare if label=="###" else image_match)
			draw.polygon(xycoords,outline=None,fill=255)
	np_match = np.array(image_match,dtype=float)/255
	np_dontcare = np.array(image_dontcare,dtype=float)/255
	np_match_dc = np.maximum(np_match, np_dontcare)
	return(np_match,np_dontcare,np_match_dc)

def load_img(imgfile):
	img = Image.open(imgfile)
	if img.mode != 'RGB':
		img = img.convert('RGB')
	iarray = np.array(img,dtype=float)
	for i in range(3):
		iarray[:,:,i] -= vgg_rgboffsets[i]
	return iarray #.transpose((2,0,1))

def convert_to_img(iarray):
	for i in range(3):
		iarray[:,:,i] += vgg_rgboffsets[i] # Remove offsets
	#img = Image.new('RGB',iarray.shape[:2])
	intarray = np.uint8(np.round(iarray))
	img = Image.fromarray(intarray)
	return img


def load_data_sample(i,image_formatstring,groundtruth_formatstring):
	xdata = load_img(image_formatstring % i)
	ydata = process_groundtruth(gt_fs % i)
	return (xdata,ydata)

def compute_overlap(imgdim,tiledim,factor):
	max_val = imgdim - tiledim
	nsteps = int(np.round(float(factor)*max_val/tiledim))
	stepsize = int(np.floor(max_val/nsteps))
	return tiledim - stepsize
	


def tile_sample(sample,name,path,tile_size=(224,224),overlap_factor=1.5,scales=[1.,0.75,0.5]):
	x,y = sample
	sx, sy = tile_size
	
	nameslist = list()

	for scale in scales:
		xs = cv2.resize(x,None,fx=scale,fy=scale)
		ys = tuple(cv2.resize(yi,None, fx=scale, fy=scale) for yi in y)
		w, h = xs.shape[:2]
		if(hasattr(overlap_factor,"__iter__")):
			overlap = overlap_factor
		else:
			overlap = (compute_overlap(w,tile_size[0], overlap_factor),compute_overlap(h,tile_size[1], overlap_factor))
		ol_x, ol_y = overlap
		for i in range(0,w-sx,sx-ol_x):
			for j in range(0,h-sy,sy-ol_y):
				tiled_x = xs[i:i+sx,j:j+sy,:]
				tiled_y = tuple(yi[i:i+sx,j:j+sy] for yi in ys)
				fullname = "%s-scale%.2f-%d-%d.npz" % (name,scale,i,j)
				np.savez("%s/%s" % (path, fullname), x=tiled_x, y_match=tiled_y[0], y_dontcare=tiled_y[1], y_fusion=tiled_y[2])
				nameslist.append(fullname)
	return nameslist
	

def gen_all_data(out_path,out_indexfile,n=1000,image_formatstring=image_fs,groundtruth_formatstring=gt_fs):
	with open(out_indexfile,'w') as ifd:
		for i in range(1,n+1):
			print("Processing data for sample #%d" % i)
			sample = load_data_sample(i,image_formatstring,groundtruth_formatstring)
			tilesnames = tile_sample(sample,"img_%d" % i, out_path)
			for name in tilesnames:
				ifd.write(name+"\n")




# Extra negative data gen


def cut_and_save_img(imgfile, outdir="tiles", overlap_factor=2.,tile_size=(224,224)):
	sx, sy = tile_size
	filebase = imgfile.split('/')[-1].split('.')[0]
	imgarray = load_img(imgfile)
	w,h = imgarray.shape[:2]
	if(hasattr(overlap_factor,"__iter__")):
		overlap = overlap_factor
	else:
		overlap = (compute_overlap(w,tile_size[0], overlap_factor),compute_overlap(h,tile_size[1], overlap_factor))
	ol_x, ol_y = overlap
	n_total = len(range(0,w-sx,sx-ol_x))*len(range(0,h-sy,sy-ol_y))
	steps = n_total/20
	print 'Starting [                    ]',
	print '\b'*22,
	sys.stdout.flush()
	k=0
	for i in range(0,w-sx,sx-ol_x):
		for j in range(0,h-sy,sy-ol_y):
			k += 1
			tile = np.copy(imgarray[i:i+sx,j:j+sy,:])
			tileimg = convert_to_img(tile)
			tileimg.save("%s/%s-tile-%d-%d.png" % (outdir,filebase,i,j))
			if(k % steps ==0):
				print '\b.',
				sys.stdout.flush()
	print '\b]  Done!'


def gen_extranegdata(dirs,outpath,use_trueneg=False):
	fileslist = list()
	for d in dirs:
		imagebase = d.split('/')[-1]
		truenegatives = (os.listdir("%s/negative" % d) if use_trueneg else [])
		falsepositives = os.listdir("%s/positive-blend" % d) #True positives should have been removed
		files = ["%s/negative/%s" % (d,f) for f in truenegatives] + ["%s/positive/%s" % (d,f) for f in falsepositives]
		for f in files:
			#Prepare data point
			filebase = f.split('/')[-1].split('.')[0]
			xdata = load_img(f)
			yzero = np.zeros(xdata.shape[:2])
			np.savez("%s/%s-%s.npz" % (outpath, imagebase, filebase), x=xdata, y_match=yzero, y_dontcare=yzero, y_fusion=yzero)
			fileslist.append("%s-%s.npz" % (imagebase, filebase))
	return fileslist



def prepare_rm_falsepos(imgfile, outdir="remove_falsepos", overlap_factor=2.,tile_size=(224,224),thres=0.1):
	sx, sy = tile_size
	filebase = imgfile.split('/')[-1].split('.')[0]
	basedir = "%s/%s" % (outdir,filebase) 
	os.mkdir(basedir)
	posdir = "%s/positive" % basedir
	posblenddir = "%s/positive-blend" % basedir
	negdir = "%s/negative" % basedir
	os.mkdir(posdir)
	os.mkdir(negdir)
	os.mkdir(posblenddir)
	imgarray = load_img(imgfile)
	imgarray_blend = load_img("%s-blend.png" % filebase)
	heatmapfile = np.load("%s-heatmap.npz" % filebase)
	heatmap = heatmapfile['arr_0']
	w,h = imgarray.shape[:2]
	if(hasattr(overlap_factor,"__iter__")):
		overlap = overlap_factor
	else:
		overlap = (compute_overlap(w,tile_size[0], overlap_factor),compute_overlap(h,tile_size[1], overlap_factor))
	ol_x, ol_y = overlap
	n_total = len(range(0,w-sx,sx-ol_x))*len(range(0,h-sy,sy-ol_y))
	steps = n_total/20
	print 'Starting [                    ]',
	print '\b'*22,
	sys.stdout.flush()
	k=0
	for i in range(0,w-sx,sx-ol_x):
		for j in range(0,h-sy,sy-ol_y):
			k += 1
			tile = np.copy(imgarray[i:i+sx,j:j+sy,:])
			tileimg = convert_to_img(tile)
			if(np.all(heatmap[i:i+sx,j:j+sy] < thres)):
				tileimg.save("%s/tile-%d-%d.png" % (negdir,i,j))
			else:
				tileimg.save("%s/tile-%d-%d.png" % (posdir,i,j))
				blendtile = np.copy(imgarray_blend[i:i+sx,j:j+sy,:])
				blendtileimg = convert_to_img(blendtile)
				blendtileimg.save("%s/tile-%d-%d.png" % (posblenddir,i,j))
			if(k % steps ==0):
				print '\b.',
				sys.stdout.flush()
	print '\b]  Done!'
	return (basedir, posblenddir)


