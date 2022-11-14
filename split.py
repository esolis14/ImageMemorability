import numpy as np
import csv
import tensorflow as tf
from numpy import mean
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import PIL
from tensorflow.keras.utils import load_img, img_to_array, array_to_img
from keras import layers, models, optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *

from loaders import load_image, load_split


# For some computational limitations I cannot selecto more than aproximately 15000 images at once.
# If possible, increase this parameter to 45000
filetrain = load_split('/Users/david/powerai-image-memorability/lamem/splits/train_1.txt')[:15000]
filetest = load_split('/Users/david/powerai-image-memorability/lamem/splits/test_1.txt')
fileval = load_split('/Users/david/powerai-image-memorability/lamem/splits/test_1.txt')[:2000]

images_x = load_image([tup[0] for tup in filetrain])
labels_y = [tup[1] for tup in filetrain]

test_images_x = load_image([tup[0] for tup in filetest])
test_labels_y = [tup[1] for tup in filetest]

val_images_x = load_image([tup[0] for tup in fileval])
val_labels_y = [tup[1] for tup in fileval]

x_train= np.array(images_x)
y_train= np.array(labels_y)

x_test= np.array(test_images_x)
y_test= np.array(test_labels_y)

x_val= np.array(test_images_x)
y_val= np.array(test_labels_y)

# We parse the data and center the data
imp = SimpleImputer(missing_values=np.NAN, strategy='mean')
#x_train = imp.fit_transform(x_train)
y_train = imp.fit_transform(y_train.reshape(-1, 1))
y_test = imp.fit_transform(y_test.reshape(-1, 1))
y_val = imp.fit_transform(y_val.reshape(-1, 1))



###################################


model = models.Sequential()
model.add(Conv2D(96, (11, 11), (4, 4), activation="relu", input_shape=(128, 128, 3)))
model.add(MaxPooling2D((3, 3), (2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(256, (5, 5), activation="relu"))
model.add(ZeroPadding2D((2, 2)))
model.add(MaxPooling2D((3, 3), (2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(384, (3, 3), activation="relu"))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(384, (3, 3), activation="relu"))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(256, (3, 3), activation="relu"))
model.add(ZeroPadding2D((1, 1)))
model.add(MaxPooling2D((3, 3), (2, 2)))
model.add(GlobalAveragePooling2D())
model.add(Dense(4096, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(4096, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1))

model.summary()

#####
# Using "Euclidean Distance" loss
def euclidean_distance_loss(y_true, y_pred):
    return tf.math.sqrt(tf.math.reduce_sum(tf.math.square(y_pred - y_true), axis=-1))
#####

model.compile(
    "adam", euclidean_distance_loss, metrics=[tf.keras.metrics.MeanAbsoluteError()])


history= model.fit(x_train, y_train, batch_size=64*8, epochs=30, verbose=1, validation_data=(x_val, y_val), workers=8, use_multiprocessing=True)

# imagestest = load_image([tup[0] for tup in filetest])
# labelstest = [tup[1] for tup in filetest]

# y_predict = model.predict(imagestest)

# print(y_predict)

print(history.history.keys())

loss = history.history['loss']
mean_absolute_error = history.history['mean_absolute_error']
val_loss = history.history['mean_absolute_error']
val_mean_absolute_error = history.history['val_mean_absolute_error']

epochs = range(1, len(loss) + 1)


# Plot accuracy
plt.plot(epochs, loss, 'b', label='Training acc')
plt.plot(epochs, val_loss, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.show()

# Plot loss
plt.plot(epochs, mean_absolute_error, 'b', label='Training loss')
plt.plot(epochs, val_mean_absolute_error, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()