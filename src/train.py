from data_loaders import load_data, parse_image
from model import define_model, VGG16_model
from show_results import plot_results


# Load data
(x_train, x_val, x_test, y_train, y_val, y_test) = load_data('../lamem/')

pre_trained = False
if pre_trained:
    model, history = VGG16_model(x_train, x_val, y_train, y_val)
    plot_results(history, 'vgg16')
    model.save('../models/vgg16_model.h5')
else:
    # Define model
    model = define_model()
    # Traing model
    history = model.fit(x_train, y_train,
                        batch_size=64*2,
                        epochs=2,
                        validation_data=(x_val, y_val),
                        workers=8,
                        use_multiprocessing=True,
                        verbose=1)
    # Plot results
    # plot_results(history, 'model')
    # Save the model
    model.save('../models/project_model.h5')

# Evaluate the model on the test data
print("Evaluation of the model on test data")
results = model.evaluate(x_test, y_test, batch_size=128)
print("Test loss, Test acc: ", results)

# Make predictions
print('PREDICT IMAGE MEMORABILITY')
img_name = input("Enter image's file name: ")
while(img_name != ''):
    img = parse_image('../' + img_name)
    print('Memorability score for ' + img_name + ': ' + str(model.predict(img)))
    img_name = input("Enter image's file name: ")