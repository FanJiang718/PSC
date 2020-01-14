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


import datetime






def get_now():
	now = datetime.datetime.now()
	tstamp = now.strftime("%d-%m-%y--%H-%M")
	return tstamp


base_data_path = "."
output_path = "./save"


vgg16_path = "%s/vgg16_weights.h5" % base_data_path
traindatapath = "%s/traindata" % base_data_path
traindataindex = "%s/trainsamples.txt" % base_data_path

weightspath = "%s/weights" % output_path
historyfile = "%s/final.gz" % output_path
logfile = "%s/log.txt" % output_path


def VGG_16(weights_path=None):
	model = Sequential()
	model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
	model.add(Convolution2D(64, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(64, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(128, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(128, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(Flatten())
	model.add(Dense(4096, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(4096, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1000, activation='softmax'))

	if weights_path:
		model.load_weights(weights_path)

	return model

def zhang_network(weights_path=vgg16_path):
	#Build the VGG16
	vgg_input = Input(shape=(3,224,224))

	x = ZeroPadding2D((1,1))(vgg_input)
	x = Convolution2D(64, 3, 3, activation='relu')(x)
	x = ZeroPadding2D((1,1))(x)
	x = Convolution2D(64, 3, 3, activation='relu')(x)
	stage1out = MaxPooling2D((2,2), strides=(2,2))(x)

	x = ZeroPadding2D((1,1))(stage1out)
	x = Convolution2D(128, 3, 3, activation='relu')(x)
	x = ZeroPadding2D((1,1))(x)
	x = Convolution2D(128, 3, 3, activation='relu')(x)
	stage2out = MaxPooling2D((2,2), strides=(2,2))(x)

	x = ZeroPadding2D((1,1))(stage2out)
	x = Convolution2D(256, 3, 3, activation='relu')(x)
	x = ZeroPadding2D((1,1))(x)
	x = Convolution2D(256, 3, 3, activation='relu')(x)
	x = ZeroPadding2D((1,1))(x)
	x = Convolution2D(256, 3, 3, activation='relu')(x)
	stage3out = MaxPooling2D((2,2), strides=(2,2))(x)

	x = ZeroPadding2D((1,1))(stage3out)
	x = Convolution2D(512, 3, 3, activation='relu')(x)
	x = ZeroPadding2D((1,1))(x)
	x = Convolution2D(512, 3, 3, activation='relu')(x)
	x = ZeroPadding2D((1,1))(x)
	x = Convolution2D(512, 3, 3, activation='relu')(x)
	stage4out = MaxPooling2D((2,2), strides=(2,2))(x)

	x = ZeroPadding2D((1,1))(stage4out)
	x = Convolution2D(512, 3, 3, activation='relu')(x)
	x = ZeroPadding2D((1,1))(x)
	x = Convolution2D(512, 3, 3, activation='relu')(x)
	x = ZeroPadding2D((1,1))(x)
	x = Convolution2D(512, 3, 3, activation='relu')(x)
	stage5out = MaxPooling2D((2,2), strides=(2,2))(x)

	x = Flatten()(stage5out)
	x = Dense(4096, activation='relu')(x)
	x = Dropout(0.5)(x)
	x = Dense(4096, activation='relu')(x)
	x = Dropout(0.5)(x)
	vggout = Dense(1000, activation='softmax')(x)

	vgg_model = Model(input=vgg_input, output=vggout)

	# Load  weights
	if weights_path:
		vgg_model.load_weights(weights_path,seq_to_functional=True)

	# Build the modified model
	#x =  Convolution2D(64, 1, 1, activation='relu')(stage1out)
	x =  Convolution2D(1, 1, 1, activation='relu')(stage1out)
	stage1deconv = UpSampling2D(size=(2, 2), dim_ordering='th')(x)

	#x =  Convolution2D(128, 1, 1, activation='relu')(stage2out)
	x =  Convolution2D(1, 1, 1, activation='relu')(stage2out)
	stage2deconv = UpSampling2D(size=(4, 4), dim_ordering='th')(x)

	#x =  Convolution2D(256, 1, 1, activation='relu')(stage3out)
	x =  Convolution2D(1, 1, 1, activation='relu')(stage3out)
	stage3deconv = UpSampling2D(size=(8, 8), dim_ordering='th')(x)

	#x =  Convolution2D(512, 1, 1, activation='relu')(stage4out)
	x =  Convolution2D(1, 1, 1, activation='relu')(stage4out)

	stage4deconv = UpSampling2D(size=(16, 16), dim_ordering='th')(x)
	#x =  Convolution2D(512, 1, 1, activation='relu')(stage5out)
	x =  Convolution2D(1, 1, 1, activation='relu')(stage5out)
	stage5deconv = UpSampling2D(size=(32, 32), dim_ordering='th')(x)

	# Merge deconvs
	merged = merge([stage1deconv,stage2deconv,stage3deconv,stage4deconv,stage5deconv],mode='concat',concat_axis=1)

	final = Convolution2D(1, 1, 1, activation='sigmoid')(merged)

	zhang_model = Model(input=vgg_input, output=final)
	
	return zhang_model


class DataSetLoader:
	def __init__(self,filelist,imagepath,shuff=True, y_ds="y_fusion"):
		self.dataset = filelist
		self.nimages = len(self.dataset)
		self.indices = range(self.nimages)
		self.basepath = imagepath
		self.mode=y_ds
		if(shuff):
			np.random.shuffle(self.indices)
		self.current_index = 0
		print("Generator for %d inputs ready" % (self.nimages))
	
	def reset_index(self):
		self.current_index=0
	
	def reshuffle(self):
		np.random.shuffle(self.indices)
	
	def get_sample(self, outer_index):
		i = outer_index % self.nimages
		archive = np.load("%s/%s" % (self.basepath, self.dataset[self.indices[i]]))
		xdata = archive["x"]
		ydata = archive[self.mode]
		return (xdata,ydata)
		
	def flow(self,batch_size = 10):
		while(True):
			xsamples = []
			ysamples = []
			thisbatchsize = min(batch_size,self.nimages-self.current_index)
			for i in range(self.current_index,self.current_index+thisbatchsize):
				x,y = self.get_sample(i)
				xsamples.append(np.array([x]))
				ysamples.append(np.expand_dims(np.array([y]),0))
			ybatch = np.vstack(ysamples)
			xbatch = np.vstack(xsamples).transpose((0,3,1,2))

			self.current_index += thisbatchsize
			if(self.current_index >= self.nimages):
				self.reshuffle()
				self.reset_index()
			yield (xbatch, ybatch)

def get_loaders(filename,traindata_dir,valid_pct,img_pct=1.):
	with open(filename,'r') as fd:
		files = [f.strip() for f in fd]
	np.random.shuffle(files)
	n_files = int(len(files)*img_pct)
	n_valid = int(n_files*valid_pct)
	valid_ds = files[:n_valid]
	train_ds = files[n_valid:n_files]
	train_loader=DataSetLoader(train_ds,traindata_dir)
	valid_loader=DataSetLoader(valid_ds,traindata_dir)
	return (train_loader, valid_loader)

class LrReducerAndSaver(Callback):
	def __init__(self, patience=0, reduce_rate=0.5, reduce_nb=20, verbose=1,reload_if_bad=False,increase_if_good=0.0,output_logfile=logfile,output_weightsdir=weightspath):
		super(Callback, self).__init__()
		self.patience = patience
		self.wait = 0
		self.best_score = np.inf
		self.reduce_rate = reduce_rate
		self.increase_if_good = increase_if_good
		self.current_reduce_nb = 0
		self.reload_if_bad = reload_if_bad
		self.reduce_nb = reduce_nb
		self.verbose = verbose
		self.now = get_now()
		self.last_weights = None
		self.weights_path = output_weightsdir
		self.logfile = output_logfile
		with open(self.logfile,'a') as logfd:
			logfd.write("time\tepoch\ttloss\tvloss\tlast_lr\n")

	def on_epoch_end(self, epoch, logs={}):
		current_score = logs.get('val_loss')
		current_trainloss = logs.get('loss')
		lr = self.model.optimizer.lr.get_value()
		with open(self.logfile,'a') as logfd:
			logfd.write("%s\t%d\t%f\t%f\t%f\n" % (get_now(),epoch,current_trainloss,current_score,lr))
		if current_score < self.best_score:
			self.best_score = current_score
			self.wait = 0
			self.last_weights = "%s/%s-weights-epoch%d.h5" % (self.weights_path, self.now, epoch)
			self.model.save_weights(self.last_weights)
			if self.verbose > 0:
				print('---current best loss: %.3f' % current_score)
			if(self.increase_if_good > 0):
				self.model.optimizer.lr.set_value(np.float32(lr*(1.+self.increase_if_good)))
		else:
			if(self.reload_if_bad and self.last_weights):
				print("Reloading weights")
				self.model.load_weights(self.last_weights)
			if self.wait >= self.patience:
				self.current_reduce_nb += 1
				if self.current_reduce_nb <= self.reduce_nb:
					self.model.optimizer.lr.set_value(np.float32(lr*self.reduce_rate))
					if self.verbose > 0:
						print("Epoch %d: reducing lr, setting to %f" % (epoch, lr*self.reduce_rate))
				else:
					if self.verbose > 0:
						print("Epoch %d: early stopping" % (epoch))
					self.model.stop_training = True
				self.wait = 0
			else:
				self.wait += 1



def train_model(valid_pct=0.1,n_epoch=100,batch_size=10,base_lr=1e-4,ds_reduction_factor=1.,patience=1,reload_if_bad=True,increase_if_good=0.05,initial_weights_path=vgg16_path,traindata_index=traindataindex, traindata_path=traindatapath, output_logfile=logfile, output_weights=weightspath):
	model = zhang_network(initial_weights_path)
	sgd = SGD(lr=(base_lr/ds_reduction_factor), momentum=0.9, nesterov=True)
	model.compile(loss='binary_crossentropy', optimizer=sgd)
	train_loader, valid_loader = get_loaders(traindata_index,traindata_path,valid_pct,ds_reduction_factor)
	gen_train = train_loader.flow(batch_size)
	gen_val = valid_loader.flow(batch_size)
	cb = LrReducerAndSaver(patience=patience,reload_if_bad=reload_if_bad,increase_if_good=increase_if_good,output_logfile=output_logfile,output_weightsdir=output_weights)
	history = model.fit_generator(gen_train,
							samples_per_epoch=train_loader.nimages,
							nb_epoch=n_epoch,
							validation_data=gen_val,nb_val_samples=valid_loader.nimages,callbacks=[cb])
	return history, train_loader, valid_loader




