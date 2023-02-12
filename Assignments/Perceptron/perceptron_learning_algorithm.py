"""

Problem: 

Create a target function f and a bidimensional dataset D and see how the perceptron learning algorithm works.
Choose a random line in the plane as your target function, where one side of theline maps to +1 and the other maps to -1. 
Choose the inputs xn of the data set as random points in theplane and evaluate the target function on each xn to get the corresponding output yn.

a) create a datset of size 100. Run the perceptronlearning algorithm on your data set and 
see how long it takes to converge and how well the final hypothesis g matches your target f.

b) plot the examples (xn, yn) as well as the target function f on a plane. Be sure to mark the examples from different classes differently.

c)  report the number of updates that the algorithm takes before it converges

d) repat evertthing for a dataset of size 1000 and compare the results with the previous case.

e) modify the algorithm such that it akes xn E R¹⁰. randomly generate lineraly seàranñe dataset of size 100 with xn E R¹⁰. How many updates does it take to converge?



"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from perceptron_ import Perceptron
def genereta_data(perceptron,size):
    X = np.random.randn(size, 2) # random inputs xn
    y = np.array([perceptron.model(x) for x in X]) # evaluate target function on each xn to get corresponding yn
    return X, y


def main(): 
    perceptron = Perceptron()
    X, y = perceptron.load_data(100)
    perceptron.fit(X, y)
    print(perceptron.accuracy(X, y))
    perceptron.draw(X, y)
  
    # second dataset
    perceptron2 = Perceptron()
    X, y = genereta_data(perceptron2, 1000)
    perceptron2.fit(X, y)
    print(perceptron2.accuracy(X, y))
    perceptron2.draw(X, y)


if __name__ == "__main__":
    main()

