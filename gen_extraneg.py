#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import sys

default_outpath = "remove_falsepos"

import dataprocessing

if __name__ == "__main__":
	print("\n")
	def usageexit():
		print("Usage: %s <path to the image dir> [--use-trueneg=false] [path to the train samples dir] [path to the train samples list]" % sys.argv[0])
		print("\n")
		sys.exit(1)

	if(len(sys.argv) < 4):
		usageexit()
	trueneg = False

	imgdir = sys.argv[1].strip()
	if(sys.argv[2].strip() == "--use-trueneg=true"):
		trueneg = True:
	if(sys.argv[2].strip() == "--use-trueneg=false"):
		trueneg = False:
	else:
		print("Second argument should be --use-trueneg=true or --use-trueneg=false, got '%s' instead" % sys.argv[2].strip())
	outdir = sys.argv[3].strip()
	outlist = sys.argv[4].strip()
	
	print "Generating and adding extra negative data points"
	flist = dataprocessing.gen_extranegdata([imgdir],outdir,use_trueneg=trueneg)
	with open(outlist,'a') as listfile:
		for f in flist:
			listfile.write(f+"\n")
	print "Done."


