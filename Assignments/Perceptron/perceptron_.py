import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class Perceptron():

    def __init__(self):
        
        self.w = np.random.randn(2)
        self.b = np.random.randn()
        self.updates = 0

    def model(self, X):
        return 1 if (np.dot(self.w, X) + self.b) >= 0 else -1
    
    def fit(self, X, y, learning_rate=0.1, max_epochs=1000):

        self.w = np.zeros(X.shape[1])
        self.b = 0

        for epoch in range(max_epochs):
            for x, target in zip(X, y):
                if (np.dot(self.w, x) + self.b) * target <= 0:
                    self.w += learning_rate * target * x
                    self.b += learning_rate * target
                    self.updates += 1
        return self.w, self.b, self.updates

    def draw(self, X, y): 
        positive_examples = X[y == 1]
        negative_examples = X[y == -1]
        plt.scatter(positive_examples[:, 0], positive_examples[:, 1], marker='+')
        plt.scatter(negative_examples[:, 0], negative_examples[:, 1], marker='_')

        x_vals = np.linspace(-3, 3, 100)
        y_vals = -(self.w[0] * x_vals + self.b) / self.w[1]
        plt.plot(x_vals, y_vals, 'r')

        plt.show()

    def accuracy(self, X, y):
        correct = 0
        for x, target in zip(X, y):
            if self.model(x) == target:
                correct += 1
        accuracy = correct / len(X)
        return accuracy

    def load_data(self, size):
        # Generate the dataset
        size = 100
        X = np.random.randn(size, 2) # random inputs xn
        y = np.array([self.model(x) for x in X]) # evaluate target function on each xn to get corresponding yn
        return X, y






    
    
