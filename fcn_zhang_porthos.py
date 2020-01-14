#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from keras.models import Sequential, Model

from keras.layers import Input, merge
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, UpSampling2D
from keras.optimizers import SGD
from keras.callbacks import Callback



import numpy as np

import gzip, cPickle as pickle


import fcn_zhang

base_data_path = "/home/i93332/src/zhang"
output_path = "/home/i93332/output"


vgg16_path = "%s/vgg16_weights.h5" % base_data_path
traindatapath = "%s/traindata" % base_data_path
traindataindex = "%s/trainsamples.txt" % base_data_path

weightspath = "%s/weights" % output_path
historyfile = "%s/final.gz" % output_path
logfile = "%s/log.txt" % output_path


if __name__ == "__main__":
	history, train_loader, valid_loader = fcn_zhang.train_model(initial_weights_path=vgg16_path,traindata_index=traindataindex, traindata_path=traindatapath, output_logfile=logfile, output_weights=weightspath)
	#history, train_loader, valid_loader = train_model(ds_reduction_factor=0.01,n_epoch=2,batch_size=1)
	history.model.save_weights("%s/weights-final-%s.h5" % (weightspath, fcn_zhang.get_now()))
	toarchive = (history.history, train_loader.dataset, valid_loader.dataset)
	with gzip.open(historyfile,'w') as fd:
		pickle.dump(toarchive, fd)
	print("All done")



