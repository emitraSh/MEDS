import numpy as np
from scipy.stats import multivariate_normal,dirichlet

def fit_GDA_MLE(x_train,y_train,):
    n = y_train.shape[0] #number of samples/rows
    x_train = x_train.reshape(n, -1) #ensures that x_train has same amount of rows # -1 means figure out number of columns automatically
    num_features = x_train.shape[1]
    classes = np.unique(y_train)
    mu = np.zeros(len(classes),num_features) #it creates a zero matrix
    sigma = np.zeros(len(classes),num_features,num_features) #zero matrix, each class gets its own squared matrix that shows the relation of each feature with each other in the specific class
    prior_mle = np.zeros(len(classes)) #zero array to stor prior probabilities-MLE

    for c in classes:
        x_c = x_train[y_train == c] #does it work only when y_train is part of x_train??
        prior_mle[c] = len(x_c) / len(x_train)

        mu[c] = np.mean(x_train[x_c,:],axis=0) # mean of each corresponding feature of that class
        sigma[c] = np.cov(x_train[x_c,:],rowvar=False) #rowvar= False shows that rows are sampled data and columns are features

    return mu,sigma,prior_mle


def prediction(x_test,classes,mu,sigma,prior_mle):
    scores = np.zeros(x_test.shape[0],mu.shape[0]) #creates a matrix that will hold the estimated class for each row?
    for c in classes:
        multivariate_normal_prob = multivariate_normal(mean=mu[c],cov=sigma[c])
        #The function """enumerate()""" is a built-in Python utility that lets you loop over a list (or any iterable) and get both the index and the item at the same time.
        for index,x in enumerate(x_test):
            scores[index,c] =np.log(prior_mle[c]) + multivariate_normal_prob.logpdf(x)

    predictions = np.argmax(scores, axis=1)
    return predictions



