import numpy as np
import csv
import tensorflow as tf
from numpy import mean
import matplotlib.pyplot as plt
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
    return numpyArray

def load_image(df):
    images = []
    j = 0
    tamano = len(df)
    for i in df:
        j = j +1
        if j % 1000 == 0:
            print(str(j)+ " of " + str(tamano) + " images processed")
        img = load_img('/Users/david/powerai-image-memorability/lamem/images/'+str(i)).resize((128,128))
        img_array = img_to_array(img)/255
        #print(img_array.shape)
        images.append(img_array)
    return images


