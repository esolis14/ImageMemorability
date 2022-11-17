# ImageMemorability
This repository contains the final project for the CS 577 Deep Learning course at the IIT, which consits of a neural network model for predicting image memorability.

For the purpose of this project the LaMem data set has been selected. This data set consists of 60,000 images, each one associated with a normalized memorability score. This data can be download from the LaMem site: http://memorability.csail.mit.edu/download.html

### How to run the code
2. Clone this repository:\
`git clone https://github.com/esolis14/ImageMemorability`

2. Dowload the dataset and extract it inside the repository's folder:\
`wget http://memorability.csail.mit.edu/lamem.tar.gz`

4. Run the `train.py` file for training and evaluating the models.
If the `pre_trained` variable is set to `True`, the Transfer Learning model is executed, Otherwise, the CNN model is built.

5.  Run the `predict.py` file for predict the memorability score of new images.

### References
Aditya Khosla, Akhil S. Raju, Antonio Torralba, and Aude Oliva. Understanding and predicting image memorability at
a large scale. In 2015 IEEE International Conference on Computer Vision (ICCV), pages 2390–2398, 2015.


