# CPP-Neural-Network

A very simply neural network (1 hidden layer of 128 neurons) built in C++ without any external libraries. Trained using MNIST dataset and weights of edges are stored in `model-neural-network.dat`. Achieved an accuracy of 96.2%! 

Training data can be found here: https://drive.google.com/file/d/1tVyvg6c1Eo5ojtiz0R17YEzcUe5cN285/view

## Features to Add
  1. Replace the sigmoid activation function with the ReLU activation function.
  2. Initialize the weights using a normal distribution with a mean of 0 and a standard deviation of 1/sqrt(number of input connections).
  3. Use a learning rate scheduler to adjust the learning rate based on the progress of training.
  4. Add L2 regularization to the loss function.
  5. Apply batch normalization to the hidden layer activations.
