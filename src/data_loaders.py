import numpy as np
from sklearn.impute import SimpleImputer
from keras.utils import load_img, img_to_array


# Load split .txt files
def load_split(path):

    d = {}
    for line in open(path, 'r'):
        split = line.strip().split(' ', 1)
        d[split[0]] = split[1]
    result = d.items()
    data = list(result)
    numpyArray = np.array(data)

    return numpyArray


# Load images
def load_images(df, path):

    images = []
    j = 0

    for i in df:
        j = j + 1
        if j % 1000 == 0:
             print(str(j)+ " of " + str(len(df)) + " images processed")
        img = load_img(path + 'images/'+str(i)).resize((128,128))
        img_array = img_to_array(img)/255
        images.append(img_array)

    return images


def parse_image(path):

    img = load_img(path, target_size=(128, 128))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)/255

    return img


# Load data from files
# Return training, validation and test features and labels
def load_data(path):

    # Due to computational limitationes, only 15000 from the training set are loaded.
    # If possible, increase this parameter to 45000
    file_train = load_split(path + 'splits/train_1.txt')[:1500]
    file_val = load_split(path + 'splits/val_1.txt')[:200]
    file_test = load_split(path + 'splits/test_1.txt')[:200]

    x_train = np.array(load_images([tup[0] for tup in file_train], path))
    y_train = np.array([tup[1] for tup in file_train])

    x_val = np.array(load_images([tup[0] for tup in file_val], path))
    y_val = np.array([tup[1] for tup in file_val])

    x_test = np.array(load_images([tup[0] for tup in file_test], path))
    y_test = np.array([tup[1] for tup in file_test])

    # Parse and center the labels
    imp = SimpleImputer(missing_values=np.NAN, strategy='mean')
    y_train = imp.fit_transform(y_train.reshape(-1, 1))
    y_test = imp.fit_transform(y_test.reshape(-1, 1))
    y_val = imp.fit_transform(y_val.reshape(-1, 1))

    return x_train, x_val, x_test, y_train, y_val, y_test


