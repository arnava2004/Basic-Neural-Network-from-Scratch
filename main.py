import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

##Let's load the MNIST dataset using tensorflow and keras (THE ONLY TIME)
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# This code is for One hot encoding for the datasets, makes life easier for results neurons
def categorical(x, n_col = None):
    if (not n_col):
        n_col = np.amax(x) + 1
    coded = np.zeros((x.shape[0], n_col))
    coded[np.arange(x.shape[0]), x] = 1
    return coded

#Converting now, y is what the image's number shows
y_test, y_train = categorical(y_test.astype("int")), categorical(y_train.astype("int"))
X_test, X_train = X_test / 255.0, X_train / 255.0

#Reshaping and confirming shapes, should be in the form of 
# (rows = number of data, columns = data itself encoded into an array)
X_test, X_train = X_test.reshape(-1, 28*28), X_train.reshape(-1, 28*28)
print(X_test.shape, X_train.shape)
inputNum = 784
outputNum = 10

# Now let's make major functions that will be used throughout the code

# The next three classes help us implement loss and activation functions

# This is the Leaky Rectified Linear Unit class, a type of activation function that is based on a ReLU, 
# but instead has a small slope for negative values
class LeakyReLU():
  def __init__(self, alpha = 0.2):
    self.alpha = alpha
  
  def __call__(self, x):
    return self.activation(x)
  
  def activation(self, x):
    return np.where(x >= 0, x, self.alpha * x)
  
  def gradient(self, x):
    return np.where(x >= 0, 1, self.alpha)

# This is the Soft Max function, used in the final layer of the neural network to convert the array of K real values 
# into one with K real values summing to one

class Softmax():
  def __init__(self): pass
  
  def __call__(self, x):
    return self.activation(x)
  
  def activation(self, x):
    e_x = np.exp(x - np.max(x, axis = -1, keepdims=True))
    return e_x / np.sum(e_x, axis= -1, keepdims = True)
  
  def gradient(self, x):
    # The error in our softmax
    p = self.activation(x)
    return p * (1 - p)


# Finally, this is the cross entropy loss class, with the purpose of being used to take the output probabilities 
# and measure the distance from the truth values

class CrossEntropy():
  def __init__(self): pass

  def loss(self, results, probabilities):
    probabilities = np.clip(probabilities, 1e-15, 1- 1e-15)
    return -results*np.log(probabilities) - (1 - results) * np.log(1- probabilities)
  
  def gradient(self, results, probabilities):
    probabilities = np.clip(probabilities, 1e-15, 1- 1e-15)
    return -(results/probabilities) + (1 - results) / (1 - probabilities)

# Now we can start to code the different layers of the network

# We first start with the Activation layer, a simple wrapper for all of the activations essentially
class Activation():
  def __init__(self, activation, name="activation"):
    self.activation = activation
    self.gradient = activation.gradient
    self.input = None
    self.output = None
    self.name = name
  
  def forward(self, x):
    self.input = x
    self.output = self.activation(x)
    return self.output
  
  def backward(self, output_error, lr = 0.01):
    return self.gradient(self.input) * output_error
  
  def __call__(self, x):
    return self.forward(x)

# This is the linear layer class, the layers that actually form the basis of the neural network we are creating
# The parameters inputN are the number of inputs (how many nuerons coming from the last layer essentially) and 
# outputN (how many neurons are in the layer next to it)
class Linear():
  def __init__(self, inputN, outputN, name="linear"):
    limit = 1 / np.sqrt(inputN)
    self.Weights = np.random.uniform(-limit, limit, (inputN, outputN))
    self.biases = np.zeros((1, outputN)) # Biases
    self.input = None
    self.output = None
    self.name = name
  
  def forward(self, x):
    self.input = x
    self.output = np.dot(self.input, self.Weights) + self.biases # Wx + b
    return self.output
  
  def backward(self, output_error, lr = 0.01):
    input_error = np.dot(output_error, self.Weights.T)
    delta = np.dot(self.input.T, output_error) # Calculate the weights error

    # Usually, we would allow an optimizer function to update the weights
    # but here, we are just using simple stochastic gradient descent

    self.Weights -= lr * delta
    self.biases -= lr * np.mean(output_error)
    return input_error
  
  def __call__(self, x):
    return self.forward(x)

# Now we can finally create the network class, using methods and classes previously defined 
# The inputs include input_dim and output_dim which are always 784 and 10 with the MNIST, and I decided to include two other
# parameters that let us decide how many neurons are in the middle two hidden layers, with names layer1 and layer2 respectively
# defaulted to 256 and 128 
class Network():
  def __init__(self, input_dim, output_dim, layer1 = 256, layer2 = 128, lr=0.01):
    # input_dim = 784, output_dim = 10 for mnist
    self.layers = [
                   Linear(input_dim, layer1, name="input"),
                   Activation(LeakyReLU(), name="relu1"),
                   Linear(layer1, layer2, name="input"),
                   Activation(LeakyReLU(), name="relu2"),
                   Linear(layer2, output_dim, name="output"),
                   Activation(Softmax(), name="softmax")
    ]
    self.lr = lr
  
  def forward(self, x):
    for layer in self.layers:
      x = layer(x)
    return x
  
  def backward(self, loss_grad):
    for layer in reversed(self.layers):
      loss_grad = layer.backward(loss_grad, self.lr)
    # Iterating backwards through the layers, backpropogation
  
  def __call__(self, x):
    return self.forward(x)

# Now reaching the final stages of the neural network, I am just making two helper methods to make the final process easier

# This function returns the accuracy of the model based on the final outputted array predicting
# what number the inputted image is MOST LIKELY (higher number, higher the chance)
def accuracy(correctOuptut, predictedOutput):
    return np.sum(correctOuptut == predictedOutput, axis = 0) / len(correctOuptut)

# This batch_loader returns the images split into different batches, default batch size being 64
# Every iteration in the for loop of each EPOCH training will handle 64 images at once 
def batch_loader(X, y = None, batch_size=64):
  n_samples = X.shape[0]
  for i in np.arange(0, n_samples, batch_size):
    # min(i + batch_size, n_samples) is just in case we reach last batch and i + 64 > total samples
    begin, end = i, min(i + batch_size, n_samples)
    if y is not None:
      yield X[begin:end], y[begin: end]
    else:
      yield X[begin:end]


# We use CrossEntropy function (results) to calculate a single probability value, we will only use the results variable to 
# assess our loss, which will proceed to be used to back propogate in the neural network model
results = CrossEntropy()

# We can edit the inner layers' neuron count by adding two more values after outputNum, but for now it is 256 and 128
neuralNetworkModel = Network(inputNum, outputNum, lr=1e-3)

# Number of EPOCHS can be changed, more EPOCHS means more iterations to train with over the same data
EPOCHS = 5

# This is the training and result of our neural network, the culmination of all the loss, activation, and layer classes we made
# Our training loop does as follows:
# 1: We do a forward pass to go through the entire neural network and get the resulting probabilities
# 2: We calculate the loss and gradient, appending it to our loss list, calculuate the accuracies and append
# 3: We get the error and with it, backpropogate and update our weights and balances
# 4: We repeat the process from steps 1-3 continuously until we run out of batches, and then print out our average loss and accuracy
# 5: We move onto our next EPOCH and start from step 1
for epoch in range(EPOCHS):
  # Arrays containing the losses and accuracies that we can append to and find the mean of when calculating averages of the 
  # losses and accuracies for every epoch
  loss = []
  accuracies = []
  for x_batch, y_batch in batch_loader(X_train, y_train):
    out = neuralNetworkModel(x_batch) # Forward pass
    loss.append(np.mean(results.loss(y_batch, out))) # Loss - For displaying
    # We just passed the inputs incorrectly.
    accuracies.append(accuracy(np.argmax(y_batch, axis=1), np.argmax(out, axis=1))) # Accuracy - For displaying
    error = results.gradient(y_batch, out) # Calculate gradient of loss
    neuralNetworkModel.backward(error) # Backpropagation
  
  print(f"Epoch {epoch + 1}, Loss: {np.mean(loss)}, Accuracy: {np.mean(accuracies)}")

out = neuralNetworkModel(X_test) # Now we run the model on the test set
print (accuracy(np.argmax(y_test, axis=1), np.argmax(out, axis=1))) # We get an accuracy of 96%


