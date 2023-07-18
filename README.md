# Hand written neural network for digit recognition (MNIST dataset)

This is a personal project regarding neural network made to better understand the math and logic behind these systems

Language used: python

Libraries used:
* tensorflow (to retrieve the MNIST data)
* numpy
* Numba
* math
* os

## Version 1
The first version (file NNv1.py) uses the simpliest scenario. A neural network consisting of two hidden layers of 32 and 16 nodes respectively
* Cost function: squared error
* Activation function for hidden layers: sigmoid
* Activation function for output: sigmoid
* Test precision: 96.52%

## Version 2
The second version (file NNv2.py) uses a neural network consisting of a single hidden layer of 128 nodes
* Cost function: cross-entropy
* Activation function for hidden layer: ReLU
* Activation function for output: softmax
* Test precision: 97.61%

I managed to get up to 97.71% test accuracy if I let the program train for 100 epoch, but it takes too much time for just a 0.1% improvement

## Manual test
I added a pygame script that let's the user draw a number using the mouse. Press a to clean the board, press s to make the neural network guess the written digit. It's hard for it to recognise every type of handwriting style since the MNIST dataset has a very small variety of handwriting styles
