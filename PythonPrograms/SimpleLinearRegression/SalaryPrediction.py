import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

X,Y = np.loadtxt("Salary_Data.csv", skiprows=1,unpack=True, delimiter=',')
plt.plot(X,Y, 'ro')

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.50,random_state=0)


plt.plot(X_train,Y_train, 'ro')

X_one = []
for item in X_train:
    X_one.append([1, item])

theta0 = theta1 = 0
theta = np.transpose(np.array([theta0, theta1]))
cost = (np.sum((np.dot(X_one, theta) - Y_train)**2))/(2*np.size(X_train))
alpha=0.05

def gradientDescent(theta0, theta1):
    theta = np.transpose(np.array([theta0, theta1]))
    temp0 = theta0 - ((alpha/np.size(X_train)) * (np.sum(np.dot(X_one, theta) - Y_train)) )
    temp1 = theta1 - ((alpha/np.size(X_train)) * np.dot((np.dot(X_one, theta) - Y_train), np.transpose(X_train)))
    theta0 = temp0
    theta1 = temp1
    return (theta0, theta1)

def costFunction(theta0, theta1):
    theta = np.transpose(np.array([theta0, theta1]))
    hypothesis = np.dot(X_one, theta)
    return (np.sum((hypothesis - Y_train)**2))/(2*np.size(X_train))

def iteration(theta0, theta1):
    (theta0, theta1) = gradientDescent(theta0, theta1)
    cost = costFunction(theta0, theta1)
    return (cost, theta0, theta1)

old_theta0 = old_theta1 = 0
for i in range (3000):
    (cost, theta0, theta1) = iteration(theta0, theta1)
    if(theta0 == old_theta0 and theta1 == old_theta1):
        break
    old_theta0 = theta0; old_theta1 = theta1
print(cost, theta0, theta1)

plt.plot(X_train,Y_train, 'bo')
x = np.linspace(1.1,10.5)
y = (theta0) + (theta1)*x
plt.plot(x, y, '-r', label='y=27275.4 + 9183.6*x')

plt.plot(X_test, Y_test, 'bo')
x = np.linspace(1.1, 10.5)
y = (theta0) + (theta1)*x
plt.plot(x, y, '-r' , label="y=27275.4 + 9183.6*x")
