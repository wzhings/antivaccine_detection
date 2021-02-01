#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun. Jan 31, 2021

An Keras(Tensorflow) implementation for the paper:
Wang, Z., Yin, Z., & Argyris, Y. (2020). Detecting Medical Misinformation on Social Media Using Multimodal Deep Learning. 
IEEE Journal of Biomedical and Health Informatics.

@author: Zuhui Wang
"""

import os, io, re, glob
import numpy as np
import matplotlib.pyplot as plt
from keras.applications import VGG19, VGG16
from keras.layers import Permute, GRU, BatchNormalization, Embedding, Dense, Lambda, Dropout, Flatten, concatenate, Bidirectional, Conv1D, LSTM, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras import Input
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau
from keras import optimizers
from datetime import datetime
from keras.models import load_model
from plot_confusion_matrix import plot_confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as PRFscore
from Generator import Generator
from SqueezeExciteLayer import channel_spatial_squeeze_excite as CSE
from SeTaAtten import SeTaAtten
import tensorflow as tf

# set root paths
mdname = 'TAMEnet-OCR-cmtAll-ocrAll-maxlen680-maxwordAll-'
ds_name = 'ds1'
sub_name = '155' 
root = '../AntivaccineProject/antivaccine-ocr/dataset0330/' + ds_name 

# Load fine-tuned fastText embeddings
vec_path = '../AntivaccineProject'
vec_file = 'metaVec-newfinetue.txt'

vec_data = os.path.join(vec_path, vec_file)
vecf = open(vec_data, 'r', encoding='utf-8', errors='ignore')
meta_vec = list(map(float, vecf.readline().split()))

# load training data
imgs_train_data = np.load(os.path.join(root, 'fea_train_' + ds_name + '_' + sub_name, ds_name + '_train_images.npy'))
cmts_train_data = np.load(os.path.join(root, 'fea_train_' + ds_name + '_' + sub_name, ds_name + '_train_comments_ocrs.npy'))
tags_train_data = np.load(os.path.join(root, 'fea_train_' + ds_name + '_' + sub_name, ds_name + '_train_tags.npy'))
# load training labels
train_labels_data = np.load(os.path.join(root, 'fea_train_' + ds_name + '_' + sub_name, ds_name + '_train_labels.npy'))

# load val data
imgs_val_data = np.load(os.path.join(root, 'fea_train_' + ds_name + '_' + sub_name, ds_name + '_val_images.npy'))
cmts_val_data = np.load(os.path.join(root, 'fea_train_' + ds_name + '_' + sub_name, ds_name + '_val_comments_ocrs.npy'))
tags_val_data = np.load(os.path.join(root, 'fea_train_' + ds_name + '_' + sub_name, ds_name + '_val_tags.npy'))
# load val labels
val_labels_data = np.load(os.path.join(root, 'fea_train_' + ds_name + '_' + sub_name, ds_name + '_val_labels.npy'))

# load testing data
imgs_test_data = np.load(os.path.join(root, 'fea_test_' + ds_name + '_' + sub_name, ds_name + '_test_images.npy'))
cmts_test_data = np.load(os.path.join(root, 'fea_test_' + ds_name + '_' + sub_name, ds_name + '_test_comments_ocrs.npy'))
tags_test_data = np.load(os.path.join(root, 'fea_test_' + ds_name + '_' + sub_name, ds_name + '_test_tags.npy'))
# load testing labels
test_labels_data = np.load(os.path.join(root, 'fea_test_' + ds_name + '_' + sub_name, ds_name + '_test_labels.npy'))

#randomly shuffle the indices
train_indices = np.arange(imgs_train_data.shape[0]) 
np.random.shuffle(train_indices)
np.random.shuffle(train_indices)

val_indices = np.arange(imgs_val_data.shape[0]) 
np.random.shuffle(val_indices)
np.random.shuffle(val_indices)

test_indices = np.arange(imgs_test_data.shape[0])
np.random.shuffle(test_indices)
np.random.shuffle(test_indices)

# shuffle training data
x_imgs_train = imgs_train_data[train_indices]
x_cmts_train = cmts_train_data[train_indices]
x_tags_train = tags_train_data[train_indices]
# shuffle val data
x_imgs_val = imgs_val_data[val_indices]
x_cmts_val = cmts_val_data[val_indices]
x_tags_val = tags_val_data[val_indices]
# shuffle testing data
x_imgs_test = imgs_test_data[test_indices]
x_cmts_test = cmts_test_data[test_indices]
x_tags_test = tags_test_data[test_indices]

# fetch label information
train_labels = []
val_labels = []
test_labels = []

for idx in range(len(train_labels_data)):
    f_class = float(train_labels_data[idx].split('_train_')[-1])
    train_labels.append(f_class)

for idx in range(len(val_labels_data)):
    f_class = float(val_labels_data[idx].split('_val_')[-1])
    val_labels.append(f_class)

for idx in range(len(test_labels_data)):
    f_class = float(test_labels_data[idx].split('_test_')[-1])
    test_labels.append(f_class)

train_lbls = np.asarray(train_labels)
val_lbls = np.asarray(val_labels)
test_lbls = np.asarray(test_labels)

y_train = train_lbls[train_indices] # get training labels
y_val = val_lbls[val_indices] # get validation labels
y_test = test_lbls[test_indices] # get testing labels

# load embedding matrix
embedding_matrix = np.load(os.path.join(root, 'fea_train_' + ds_name + '_' + sub_name, ds_name + '_embeddings.npy'))

# load hyper parameters 
par = np.load(os.path.join(root, 'fea_train_' + ds_name + '_' + sub_name, ds_name + '_hyper_parameters.npy'), allow_pickle=True)
maxlen = par.item().get('maxlen')
embedding_dim = par.item().get('embedding_dim')
num_words = par.item().get('num_words')

# Build multi-modal architecture
## For image channel
# load vgg19 fine-tuned model
modelpath = '../AntivaccineProject/'
modelfile = 'VGG19-Image-Only-FT.13-0.84-ds4-20191205-170956.hdf5' # new saved model: don't change layer name!
smodel = os.path.join(modelpath, modelfile)
ft_model = load_model(smodel)
x = CSE(ft_model.layers[-9].get_output_at(-1))
encoded_imgs_gap = GlobalAveragePooling2D()(x) 
encoded_imgs = Dense(128, activation='relu')(encoded_imgs_gap) 

for layer in ft_model.layers:
    layer.trainable = False

##For comment (caption) channel
cmts_input = Input(shape=(maxlen,))
cmts_layer1 = Embedding(num_words, embedding_dim, weights=[embedding_matrix], trainable=True)(cmts_input)
cmts_layer2 = Bidirectional(GRU(64, return_sequences=True, dropout=0.25, recurrent_dropout=0.25))(cmts_layer1)
encoded_cmts = SeTaAtten(attention_dim=maxlen, meta_vec=meta_vec)(cmts_layer2)

##For Hashtag channel
#For tag channel [B, H, W, C], (batch, new_rows, new_cols, filters) if data_format is "channels_last"
tags_input = Input(shape=(maxlen,))
tags_layer1 = Embedding(num_words, embedding_dim, weights=[embedding_matrix], trainable=True)(tags_input)
tags_layer2 = Dense(128, activation='tanh')(tags_layer1) 
encoded_tags = SeTaAtten(attention_dim=maxlen, meta_vec=meta_vec)(tags_layer2)


#For projection function: project images and tags into comment same space
extend = Lambda(lambda x: tf.reshape(x, (-1, 1, 128)))
encoded_cmts_front = extend(encoded_cmts)
encoded_tags_front = extend(encoded_tags)

# A project module
share_input = Input(shape=(1, 128))
share_output = Dense(128, activation='relu')(share_input)
project_module = Model(share_input, share_output)
p_encoded_cmts = project_module(encoded_cmts_front)
p_encoded_tags = project_module(encoded_tags_front)


encoded_imgs_ex = extend(encoded_imgs) 
encoded_cmts_ex = extend(p_encoded_cmts) 
encoded_tags_ex = extend(p_encoded_tags) 
concatenated4 = concatenate([encoded_imgs_ex, encoded_cmts_ex, encoded_tags_ex], axis=1) 

# three-branch summation fusion with attention
concatenated4_fusion = SeTaAtten(attention_dim=maxlen, meta_vec=meta_vec)(concatenated4) 
concatenated_new = concatenate([encoded_imgs_gap, encoded_cmts, encoded_tags, concatenated4_fusion], axis=-1) 

x = Dense(256, activation='relu')(concatenated_new)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(64, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
answer = Dense(1, activation='sigmoid')(x)
## The End of the model architecture ##

timestp = datetime.now().strftime('%Y%m%d-%H%M%S')
cmodel = Model([ft_model.input, cmts_input, tags_input], answer, name='TAME-generator')    
cmodel.compile(optimizer=optimizers.RMSprop(lr=1e-4), loss='binary_crossentropy', metrics=['acc']) #Adam
cmodel.name = mdname
cmodel.summary()

#save model as .hdf5 files
if not os.path.isdir('./weights'):
    os.mkdir('./weights')
mdpath = './weights/' + cmodel.name + '-' + ds_name + '-' + timestp
if not os.path.exists(mdpath):
    os.mkdir(mdpath)
mdname = cmodel.name + '-{epoch:02d}-{val_acc:.2f}-' + ds_name + '-' + timestp + '.hdf5'
mdfile = os.path.join(mdpath, mdname)
ckpt = ModelCheckpoint(mdfile, monitor='val_acc', verbose=0, save_best_only=True, mode='max') #save whole model
early = EarlyStopping(monitor='val_loss', mode='min', patience=20) # prevent overfitting

# save log files
if not os.path.isdir('./logs'):
    os.mkdir('./logs')
logpath = './logs/' + cmodel.name + '-' + ds_name + '-' + timestp
if not os.path.isdir(logpath):
    os.mkdir(logpath)
logname = cmodel.name + '-' + ds_name + '-' + timestp + '.log'
logfile = os.path.join(logpath, logname)
csv_logger = CSVLogger(logfile, append=True)
reduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5)
callbacks_list = [ckpt, csv_logger, reduceLR]

bat_size = 30 
# create train, val, test generators
train_data_gen = Generator([x_imgs_train, x_cmts_train, x_tags_train], y_train, batch_size=bat_size)
val_data_gen = Generator([x_imgs_val, x_cmts_val, x_tags_val], y_val, batch_size=bat_size)
test_data_gen = Generator([x_imgs_test, x_cmts_test, x_tags_test], y_test, batch_size=bat_size)

# to compute number of steps for each epoch
num_train_samples = len(x_imgs_train)
# Build model fit generator
history = cmodel.fit_generator(generator = train_data_gen, epochs=100, steps_per_epoch=(num_train_samples // bat_size), validation_data = val_data_gen, verbose=1, workers=8, callbacks=callbacks_list)

# compute the testing accuracy
pred_f_ds1 = cmodel.predict_generator(generator=test_data_gen)
pred_ds1 = np.around(pred_f_ds1)
pred_ds1 = list(map(float, pred_ds1))
y_temp_ds1 = list(map(float, y_test))
acc1_ds1 = np.sum(np.equal(pred_ds1, y_temp_ds1)) / len(y_temp_ds1)
print ('The Test testing accuracy is: ', acc1_ds1)
