import numpy as np
from sklearn.metrics import accuracy_score

class logestic_regression():
    def __init__(self):
        self.losses = [] #stores losses at each training step/ to track performance
        self.train_accuracy = [] #tracks accuracy per epoch

    def fit(self,x,y, epochs):
        x = self._transformed_x(x) #what is happening????
        y = self._transdormed_y(y)

        self.weights = np.zeros(x.shape[1]) #each feature should have a corresponding weight??
        self.bias = 0
        for i in range(epochs):
            x_dot_weights = np.matmul(self.weights, x.transpose()) +self.bias #matmul: matrix multiplication
            pred = self._sigmoid(x_dot_weights)


    def _sigmoid(self,x):
        return np.array([self._sigmoid_function(value) for value in x])
    def _sigmoid_function(self,x):
        if x >= 0:
            z = np.exp(-x)
            return 1 / (1 + z)
        else:
            z = np.exp(x)
            return z / (1 + z)

#to be continued...https://github.com/casper-hansen/Logistic-Regression-From-Scratch/blob/main/src/logistic_regression/model.py


