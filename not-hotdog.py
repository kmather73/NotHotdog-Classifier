import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

import tensorflow as tf
from skimage import exposure
from tensorflow.contrib.layers import flatten
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from tensorflow.contrib.layers import flatten
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

import cv2
import glob

def rotateImage(img, angle):
    (rows, cols, ch) = img.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2), angle, 1)
    return cv2.warpAffine(img, M, (cols,rows))
    
    
def loadBlurImg(path, imgSize):
    img = cv2.imread(path)
    #angle = np.random.randint(-180, 180)
    #img = rotateImage(img, angle)
    img = cv2.blur(img,(5,5))
    img = cv2.resize(img, imgSize)
    return img

def loadImgClass(classPath, classLable, classSize, imgSize):
    x = []
    y = []
    
    for path in classPath:
        img = loadBlurImg(path, imgSize)        
        x.append(img)
        y.append(classLable)
        
    while len(x) < classSize:
        randIdx = np.random.randint(0, len(classPath))
        img = loadBlurImg(classPath[randIdx], imgSize)
        x.append(img)
        y.append(classLable)
        
    return x, y

def loadData(img_size, classSize):
    hotdogs = glob.glob('./hotdog/**/*.jpg', recursive=True)
    notHotdogs = glob.glob('./not-hotdog/**/*.jpg', recursive=True)
    
    
    imgSize = (img_size, img_size)
    xHotdog, yHotdog = loadImgClass(hotdogs, 0, classSize, imgSize)
    xNotHotdog, yNotHotdog = loadImgClass(notHotdogs, 1, classSize, imgSize)
    print("There are", len(xHotdog), "hotdog images")
    print("There are", len(xNotHotdog), "not hotdog images")
    
    X = np.array(xHotdog + xNotHotdog)
    y = np.array(yHotdog + yNotHotdog)
    #y = y.reshape(y.shape + (1,))
    return X, y


def buildNetwork(X, keepProb):
    mu = 0
    sigma = 0.3
    
    output_depth = {
        0 : 3,
        1 : 8,
        2 : 16,
        3 : 32,
        4 : 3200,
        5 : 240,
        6 : 120, 
        7 : 43,
    }
    
    #Layer 1: Convolutional + MaxPooling + ReLu + dropout. Input = 64x64x3. Output = 30x30x8.
    layer_1 = tf.Variable( tf.truncated_normal([5,5,output_depth[0],output_depth[1]], mean=mu, stddev=sigma))
    layer_1 = tf.nn.conv2d(X, filter=layer_1, strides=[1,1,1,1], padding ='VALID')
    layer_1 = tf.add(layer_1, tf.zeros(output_depth[1]))
    layer_1 = tf.nn.max_pool(layer_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    layer_1 = tf.nn.dropout(layer_1, keepProb)
    layer_1 = tf.nn.relu(layer_1)
    
    return layer_1



def toGray(images):
    # rgb2gray converts RGB values to grayscale values by forming a weighted sum of the R, G, and B components:
    # 0.2989 * R + 0.5870 * G + 0.1140 * B 
    # source: https://www.mathworks.com/help/matlab/ref/rgb2gray.html
    
    images = 0.2989*images[:,:,:,0] + 0.5870*images[:,:,:,1] + 0.1140*images[:,:,:,2]
    return images

def normalizeImages(images):
    # use Histogram equalization to get a better range
    # source http://scikit-image.org/docs/dev/api/skimage.exposure.html#skimage.exposure.equalize_hist
    images = (images / 255.).astype(np.float32)
    
    for i in range(images.shape[0]):
        images[i] = exposure.equalize_hist(images[i])
    
    images = images.reshape(images.shape + (1,)) 
    return images

def preprocessData(images):
    grayImages = toGray(images)
    return normalizeImages(grayImages)

def normalizeImages2(images):
    for i in range(images.shape[0]):
        cv2.normalize(images[i],images[i], alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    
    # if convert to gray scale use this after
    print("images has shape before", images.shape)
    #images = images.reshape(images.shape + (1,)) 
    #print("images has shape after", images.shape)
    return images


def karasModel(inputShape):
    model = Sequential()
    model.add(Convolution2D(8, 5, 5, border_mode='valid', input_shape=inputShape))
    #model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))

    model.add(Convolution2D(16, 3, 3))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))

    model.add(Convolution2D(32, 3, 3))
    #model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dense(240))

    model.add(Activation('relu'))
    model.add(Dense(120))

    #model.add(Activation('relu'))
    model.add(Dense(2))

    model.add(Activation('softmax'))
    return model 
   
size = 64
classSize = 2000
scaled_X, y = loadData(size, classSize)

n_classes = len(np.unique(y))
print("Number of classes =", n_classes)

scaled_X = preprocessData(scaled_X)
#scaled_X = normalizeImages(scaled_X)
label_binarizer = LabelBinarizer()

#y = label_binarizer.fit_transform(y)
from keras.utils.np_utils import to_categorical
y = to_categorical(y)
print("y shape", y.shape)
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)

print("train shape X", X_train.shape)
print("train shape y", y_train.shape)
inputShape = (size, size, 1)
model = karasModel(inputShape)

#y_one_hot = label_binarizer.fit_transform(y_train)
#y_one_hoy = tf.one_hot(y_train, 2)
print("train shape y", y.shape)
model.compile('adam', 'categorical_crossentropy', ['accuracy'])
history = model.fit(X_train, y_train, nb_epoch=5, validation_split=0.1)

y_one_hot_test = label_binarizer.fit_transform(y_test)

metrics = model.evaluate(X_test, y_test)
for metric_i in range(len(model.metrics_names)):
    metric_name = model.metrics_names[metric_i]
    metric_value = metrics[metric_i]
    print('{}: {}'.format(metric_name, metric_value))



