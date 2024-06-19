import numpy as np
import os
from keras_preprocessing.image import load_img, img_to_array
import csv
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import pylab
from tensorflow.keras.layers import Conv2D, MaxPooling2D,Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
# from tensorflow.python.keras.optimizers import SGD    # Eager Execution 啟動 改為 tf.keras
from tensorflow.keras.activations import relu

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import SGD

from tensorflow.keras.applications.vgg16 import VGG16

vgg16 = VGG16(weights='imagenet', include_top=False)
vgg16.summary()

data_list = os.listdir('/home/fagc2267/PycharmProjects/Triplet_loss/DDT_Mnist/5x4_Mnist/img')
imgs = []
for name in data_list:
    img = cv2.imread(os.path.join('/home/fagc2267/PycharmProjects/Triplet_loss/DDT_Mnist/5x4_Mnist/img/' + name))
    imgs.append(img)
model = Model(inputs=vgg16.input, outputs=vgg16.get_layer('block4_pool').output)

output = model.predict(imgs)

print(output.shape)