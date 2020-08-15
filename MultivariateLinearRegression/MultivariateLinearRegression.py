import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("FuelConsumptionCo2.csv")

df.head()
a=0.01

x = df[['ENGINESIZE','CYLINDERS']].values
y = df[['CO2EMISSIONS']].values
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=0)

X = []
for row in x_train:
    r = [1]
    for item in row:
        r.append(item)
    X.append(r)

X = np.asmatrix(X)

theta = np.zeros(((X[0].size), 1))


fig = plt.figure()
ax = plt.axes(projection = '3d')
zs = df['CO2EMISSIONS']
xs=df['ENGINESIZE']
ys=df['CYLINDERS']
ax.scatter3D(xs, ys, zs, 'red')

Y = y_train

h = np.dot(X, theta)

cost = np.sum (np.dot(np.transpose(h-Y), (h-Y)))*(1/(2*X.shape[0]))
temp = np.zeros(theta.shape)

def gradientDescent(theta, X):
    h = np.dot(X, theta)
    cost = np.sum(np.dot(np.transpose(h-Y), (h-Y)))*(1/(2*X.shape[0]))
    print(cost)
    for i in range(0, theta.shape[0]):
        temp[i] = theta[i] - np.sum(np.dot((h-Y), X[i])) * (a/X.shape[0])
    for i in range(0, theta.shape[0]):
        theta[i] = temp[i]
    return(theta, X, cost)

oldCost = 0
for i in range(0, 3000):
    (theta, X, cost) = gradientDescent(theta, X)
    if((oldCost - cost) **2 < 0.0000000002):
        break
print(oldCost)
print(cost)
print(theta)

