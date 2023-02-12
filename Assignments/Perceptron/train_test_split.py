"""
ROBLEM
Estimate the bias and variance for a model over multiple bootstrap samples:

a) Loads the Session 4 dataset:
b) Split it into training and test
c) Estimate the mean squared error (MSE) for a perceptron as well as the biasand variance for the model error over 200 bootstrap samples.

"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# load the dataset

data = pd.read_csv('SL_COURSE_IE/Assignments/Perceptron/session_4_glass.csv')

print(data)

# split the dataset into training and test

from sklearn.model_selection import train_test_split

X = data.drop('Type', axis=1)
y = data['Type']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# estimate the mean squared error (MSE) for a perceptron as well as the bias and variance for the model error over 200 bootstrap samples.

from sklearn.linear_model import Perceptron

perceptron = Perceptron()

perceptron.fit(X_train, y_train)

y_pred = perceptron.predict(X_test)

from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, y_pred)

print('MSE: ', mse)

# bias and variance for the model error over 200 bootstrap samples.

from sklearn.utils import resample

def bootstrap(X, y, n=200):
    X_bootstrap = []
    y_bootstrap = []
    for i in range(n):
        X_bootstrap.append(resample(X))
        y_bootstrap.append(resample(y))
    return X_bootstrap, y_bootstrap

X_bootstrap, y_bootstrap = bootstrap(X_train, y_train)

def bias_variance(X, y, n=200):
    bias = []
    variance = []
    for i in range(n):
        perceptron.fit(X[i], y[i])
        y_pred = perceptron.predict(X_test)
        bias.append(mean_squared_error(y_test, y_pred))
        variance.append(np.var(y_pred))
    return bias, variance

bias, variance = bias_variance(X_bootstrap, y_bootstrap)

print('Bias: ', np.mean(bias))
print('Variance: ', np.mean(variance))

# Plot the bias and variance

plt.plot(bias, label='bias')
plt.plot(variance, label='variance')
plt.legend()
plt.show()

# Plot the bias and variance

plt.plot(bias, label='bias')
plt.plot(variance, label='variance')
plt.legend()
plt.show()

