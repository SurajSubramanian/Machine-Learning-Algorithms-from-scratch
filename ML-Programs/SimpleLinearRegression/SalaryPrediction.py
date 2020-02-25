# The following code implements Simple Linear Regression from scratch.
# The motive is to get a good grasp of basic concepts like computing the cost function, implementing gradient descent and 
# using vector notations. It is suggested that you watch Andrew NGs Lectures on Linear Regression and also try out his 
# exercises in Octave before going through this code !

# importing necessary packages - numpy: for representing vectors, pandas: for reading csv files and matplotlib: for plotting 
# our dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# loading dataset
X,Y = np.loadtxt("Salary_Data.csv", skiprows=1,unpack=True, delimiter=',')
plt.plot(X,Y, 'ro')

# splitting dataset into train and test parts
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.50,random_state=0)

plt.plot(X_train,Y_train, 'ro')

# Insert 1 to the start of each row in the input matrix as hypothesis h(x) = w0x0 + w1x1 where x0=1
X_one = []
for item in X_train:
    X_one.append([1, item])

# We have 2 parameters to learn : theta0 and theta1
theta0 = theta1 = 0
theta = np.transpose(np.array([theta0, theta1]))
cost = (np.sum((np.dot(X_one, theta) - Y_train)**2))/(2*np.size(X_train))
alpha=0.05


def gradientDescent(theta0, theta1):
    # simultaneously updating theta0 and theta1
    theta = np.transpose(np.array([theta0, theta1]))
    temp0 = theta0 - ((alpha/np.size(X_train)) * (np.sum(np.dot(X_one, theta) - Y_train)) )
    temp1 = theta1 - ((alpha/np.size(X_train)) * np.dot((np.dot(X_one, theta) - Y_train), np.transpose(X_train)))
    theta0 = temp0
    theta1 = temp1
    return (theta0, theta1)

def costFunction(theta0, theta1):
    # returns the cost function for the given value of theta
    theta = np.transpose(np.array([theta0, theta1]))
    hypothesis = np.dot(X_one, theta)
    return (np.sum((hypothesis - Y_train)**2))/(2*np.size(X_train))

def iteration(theta0, theta1):
    # Implements a single iteration of gradient descent and computes the cost
    (theta0, theta1) = gradientDescent(theta0, theta1)
    cost = costFunction(theta0, theta1)
    return (cost, theta0, theta1)

old_theta0 = old_theta1 = 0
# Keep learning till there is no change in the value of theta between previous and current iteration
for i in range (3000):
    (cost, theta0, theta1) = iteration(theta0, theta1)
    if(theta0 == old_theta0 and theta1 == old_theta1):
        break
    old_theta0 = theta0; old_theta1 = theta1
print(cost, theta0, theta1)

# plotting our line through the training data
plt.plot(X_train,Y_train, 'bo')
x = np.linspace(1.1,10.5)
y = (theta0) + (theta1)*x
plt.plot(x, y, '-r', label='y={} + {}x'.format(theta0, theta1))

# plottling our line through the test data
plt.plot(X_test, Y_test, 'bo')
x = np.linspace(1.1, 10.5)
y = (theta0) + (theta1)*x
plt.plot(x, y, '-r' , label='y={} + {}x'.format(theta0, theta1))
