import pprint

import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import pprint
import pickle


#Cost function = binary cross_entropy loss function = measures the error iteratively
#Backward pass/ Gradiant computation = computing the gradiant of cost function to update the model step by step
class LogisticRegression:
    def __init__(self,learning_rate=0.001):
        np.random.seed(22)
        self.learning_rate = learning_rate

    def fit(self,X,y,iteration_num):
        self.X = X
        self.y = y
        #Initialize w and b
        self.W = np.zeros(self.X.shape[1])
        self.b = 0.0
        costs = []
        for i in range(iteration_num):
            sigmoid_prediction = self.forward(self.X , self.W , self.b)

            cost = self.compute_cost(sigmoid_prediction)
            costs.append(cost)

            self.compute_gradient(sigmoid_prediction)

            self.W = self.W - self.learning_rate * self.dW
            self.b = self.b - self.learning_rate * self.dbias_term

            # print cost every 100 iterations
            if i % 10000 == 0:
                print("Cost after iteration {}: {}".format(i, cost))

        #if plot_cost:
        fig = px.line(y=costs, title="Cost vs Iteration", template="plotly_dark")
        fig.update_layout(
            title_font_color="#41BEE9",
            xaxis=dict(color="#41BEE9", title="Iterations"),
            yaxis=dict(color="#41BEE9", title="cost")
        )
        fig.show()

    def forward(self,X, W = None , b = None):
        if W is None:
            W = self.W
        if b is None:
            b = self.b
        z = np.matmul(X, W) + b
        sig_res = self.sigmoid(z)
        return sig_res

    def sigmoid(self,z):
        return 1/(1+np.exp(-z))

    def compute_cost(self, sigmoid_prediction):
        m = self.X.shape[0]
        cost_func = np.sum((-np.log(sigmoid_prediction + 1e-8) * self.y) + (- np.log(1 - sigmoid_prediction + 1e-8))* (1- self.y))
        #we are adding small value epsilon to avoid log of 0
        cost = cost_func / m
        return cost

    def compute_gradient(self, sigmoid_prediction):
        m = self.X.shape[0]
        self.dW = np.matmul(self.X.T, (sigmoid_prediction - self.y)) #res -> sum of contribution per feature, each feature has a gradient vector in dW
        self.dWm = np.array([np.mean(grad) for grad in self.dW])
        #to use average gradient per feature instead of sum makes the learning rate independent of dataset size?? what?

        #bias term:
        self.dbias_term = np.sum(np.subtract(sigmoid_prediction, self.y))

    def predict(self, X , W = None , b = None):
        if W is None:
            W = self.W
        if b is None:
            b = self.b
        predicted_y = self.forward(X , W , b)
        return np.round(predicted_y)

    def save_model(self,filename= 'Name'):
        model_data ={
            'learning_rate': self.learning_rate,
            'W': self.W,
            'b': self.b,
        }
        with open(filename, 'wb') as file:
            pickle.dump(model_data, file)


    def load_model(cls, filename):
        with open(filename, 'rb') as file:
            model_data = pickle.load(file)

        # Create a new instance of the class and initialize it with the loaded parameters
        loaded_model = cls(model_data['learning_rate'])
        loaded_model.W = model_data['W']
        loaded_model.b = model_data['b']

        return loaded_model








