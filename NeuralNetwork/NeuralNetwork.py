# Only library used !
import numpy as np
# Single function used to calculate both the sigmoid and the derivative of the sigmoid 
# ( deriv=False -> sigmoid(x), derive=True -> derivative(sigmoid(x)) )
def sigmoid(x, deriv=False):
    if(deriv==True):
        return sigmoid(x)*sigmoid(1-x)
    else:
        return 1/(1 + np.exp(-x))
    
# simple data
X = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
y = np.array([[0],[1],[0],[1]])

np.random.seed(1)
# creating 2 weight vectors and 2 bias vectors
w0 = 2*np.random.random((3,4)) - 1
w1 = 2*np.random.random((4,1)) - 1

b0 = 2*np.random.random((1,4)) - 1
b1 = 2*np.random.random((1,1)) - 1

# Training
for j in range(60000):
    # Forward Propagation
    layer0 = X
    layer1 = sigmoid(np.dot(layer0, w0))
    layer2 = sigmoid(np.dot(layer1, w1))
    # Backward Propagation
    # Output Error (dC / da)
    l2_error = y - layer2
    if(j%1000 == 0):
        print( "Error :", str(np.mean(np.sum(l2_error**2))))
    l2_delta = np.multiply(l2_error, sigmoid(layer2, deriv=True))

l1_error = np.dot(l2_delta, w1.T)
l1_delta = np.multiply(l1_error, sigmoid(layer1, deriv=True))
# Updation of weights
w1 += np.dot(layer1.T, l2_delta)
w0 += np.dot(layer0.T, l1_delta)
# function used for testing and evaluation !
def evaluate(input):
    layer0 = input
    layer1 = sigmoid(np.dot(layer0, w0))
    layer2 = sigmoid(np.dot(layer1,w1))
    return layer2


