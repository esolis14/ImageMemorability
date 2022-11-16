from keras import layers, models, optimizers
from keras.applications.vgg16 import VGG16
from keras.metrics import MeanAbsoluteError

from show_results import plot_results

def define_model():

    model = models.Sequential()
    model.add(layers.Conv2D(96, (11, 11), (4, 4), activation="relu", input_shape=(128, 128, 3)))
    model.add(layers.MaxPooling2D((3, 3), (2, 2)))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(256, (5, 5), activation="relu"))
    model.add(layers.ZeroPadding2D((2, 2)))
    model.add(layers.MaxPooling2D((3, 3), (2, 2)))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(384, (3, 3), activation="relu"))
    model.add(layers.ZeroPadding2D((1, 1)))
    model.add(layers.Conv2D(384, (3, 3), activation="relu"))
    model.add(layers.ZeroPadding2D((1, 1)))
    model.add(layers.Conv2D(256, (3, 3), activation="relu"))
    model.add(layers.ZeroPadding2D((1, 1)))
    model.add(layers.MaxPooling2D((3, 3), (2, 2)))
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(4096, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(4096, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1))

    model.compile(loss='mse',
            optimizer=optimizers.Adam(learning_rate=1e-4),
            metrics=[MeanAbsoluteError()])

    model.summary()

    return model


def VGG16_model(x_train, x_val, y_train, y_val):
    input = layers.Input(shape=(128, 128, 3)) 
    conv_base = VGG16(weights='imagenet', include_top=False, input_tensor=input)

    # Freeze the base
    conv_base.trainable = False

    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1))

    # Compile the model
    model.compile(loss='mse',
                optimizer=optimizers.Adam(learning_rate=1e-4),
                metrics=[MeanAbsoluteError()])
    
    # Train the model with the frozen base
    history_frozen = model.fit(x_train, y_train,
                    batch_size=64*2,
                    epochs=30,
                    validation_data=(x_val, y_val),
                    workers=8,
                    use_multiprocessing=True,
                    verbose=1)

    plot_results(history_frozen, 'vgg16_intermediate')

    # Unfreeze the base
    conv_base.trainable = True

    # Recompile the model
    model.compile(loss='mse',
            optimizer=optimizers.Adam(learning_rate=1e-4),
            metrics=[MeanAbsoluteError()])
    
    # Train the whole network
    history = model.fit(x_train, y_train,
                    batch_size=64*2,
                    epochs=5,
                    validation_data=(x_val, y_val),
                    workers=8,
                    use_multiprocessing=True,
                    verbose=1)

    model.summary()

    return model, history