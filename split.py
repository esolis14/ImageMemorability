import numpy as np
import csv
import tensorflow as tf
from numpy import mean


import PIL
# example of converting an image with the Keras API
from tensorflow.keras.utils import load_img, img_to_array, array_to_img

def load_split(path):
    d={}
    for line in open(path,'r'):
        split = line.strip().split(' ', 1)
        d[split[0]] = split[1]
    #print(d)
    result = d.items()
    data = list(result)
    #print(data)
    numpyArray = np.array(data)
    #print(numpyArray)
    return data



# Get images paths
#result = [tup[0] for tup in numpyArray]
 # Get labels
#result = [tup[1] for tup in numpyArray]

file = load_split('/Users/david/powerai-image-memorability/lamem/splits/train_1.txt')[:15000]


def load_image(df):
    images = []
    j = 0
    tamano = len(df)
    for i in df:
        j = j +1
        if j % 100 == 0:
            print(str(j)+ " of " + str(tamano) + " images processed")
        img = load_img('/Users/david/powerai-image-memorability/lamem/images/'+str(i)).resize((128,128))
        img_array = img_to_array(img)/255
        #print(img_array.shape)
        images.append(img_array)
    return images

images = load_image([tup[0] for tup in file])
labels = [tup[0] for tup in file]


###################################
from keras import layers
from keras import models
from keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *

model = models.Sequential()
model.add(Conv2D(96, (11, 11), (4, 4), activation="relu", input_shape=(128, 128, 3)))
model.add(MaxPooling2D((3, 3), (2, 2)))
model.add(BatchNormalization())
model.add(Dense(4096, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1))

model.summary()


model.compile(
    loss='binary_crossentropy',
    optimizer=optimizers.RMSprop(learning_rate=1e-4),
    metrics=['acc'])

model.fit(np.array(images), np.array(labels), batch_size=64, validation_split=0.10)

