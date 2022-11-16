import matplotlib.pyplot as plt

def plot_results(history, filename):

    epochs = range(1, len(history.history['loss']) + 1)

    # Loss
    plt.subplot(211)
    plt.plot(epochs, history.history['loss'], 'b--', label='Training loss')
    plt.plot(epochs, history.history['val_loss'], 'r', label='Validaton loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and validation loss')
    plt.legend()
    
     # MAE
    plt.subplot(212)
    plt.plot(epochs, history.history['mean_absolute_error'], 'b--', label='Training MAE')
    plt.plot(epochs, history.history['val_mean_absolute_error'], 'r', label='Validation MAE')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.title('Training and validation MAE')
    plt.legend()
    plt.show()
    
    plt.tight_layout()

    # Save the plot
    plt.savefig('../plots/' + filename + '_results.png')