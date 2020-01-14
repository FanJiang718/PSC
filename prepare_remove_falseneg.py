#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import sys

default_outpath = "remove_falsepos"

import dataprocessing

if __name__ == "__main__":
	print("\n")
	def usageexit():
		print("Usage: %s base_image_file [output dir]" % sys.argv[0])
		print("\n")
		sys.exit(1)

	if(len(sys.argv) < 2):
		usageexit()

	baseimg = sys.argv[1].strip()
	print("Processing image %s (and npz and blend)" % baseimg)

	if(len(sys.argv) < 3):
		print("Using default output path %s" % default_outpath)
		outpath = default_outpath
	else:
		outpath = sys.argv[2].strip()
		print("Using out path %s" % outpath)
	
	print "We will tile image %s for manual processing" % baseimg
	basedir, posblenddir = dataprocessing.prepare_rm_falsepos(baseimg,outpath)
	print "\n"
	print "Done. You should now\n1) Go to %s and delete all of the pictures containing *true positives* (text). When in doubt delete too many rather than not enough pictures.\n2) Run:\n  python gen_extraneg.py %s [path to the training dataset dir] [path to the train samples list]" % (posblenddir, basedir)


