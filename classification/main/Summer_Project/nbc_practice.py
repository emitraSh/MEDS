#Summary of NBC: Classifying the target of a dataset in which, features are considered to be mutually independent.
# we want to calculate the probability of Class C given Data= (P(C):prior/probability of C)*(multP(xi|C):likelihood)
#basefd on data-type of each feature we have different distribution/likelihoods, and prior can be calculated by MLE or Bayesian


#Chat_GPT
from collections import defaultdict
#creates default values for unfamiliar keys
import numpy as np

class NaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y) #finds unique items in an array of data, it uses Dirichlet by default?
        #calculating the priors by MLE:
        self.class_probs = {c: np.mean(y == c) for c in self.classes} #it creates a dictionary for classes and their values are their probability
        #another way:
        """self.class_probs = {}
        self.feature_probs[c] = {}
        for c in self.classes:
            X_c = X[y == c]
            self.class_probs[c] = len(X_c) / len(X)
            self.feature_probs[c] = {}"""

        self.feature_probs = defaultdict(lambda: defaultdict(lambda: 1))  # Laplace smoothing
        # it creates a nested dictionary that shows probability of each feature for each class/components of Likelihood

        for c in self.classes:
            X_c = X[y == c]
            for i in range(X.shape[1]): # in numpy shape(data,rows) gives the range of each dimension
                values, counts = np.unique(X_c[:, i], return_counts=True)
                for v, count in zip(values, counts):
                    self.feature_probs[i][(v, c)] = (count + 1) / (len(X_c) + len(values))

    def predict(self, X):
        preds = []
        for x in X:
            scores = {}
            for c in self.classes:
                score = np.log(self.class_probs[c])
                for i, val in enumerate(x):
                    score += np.log(self.feature_probs[i][(val, c)])
                scores[c] = score
            preds.append(max(scores, key=scores.get))
        return np.array(preds)



#Youtube
df =[]
y = [1,0,1,1,0]
X = [[2,3,4,2,3],[2,3,4,5,6]]
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
nbc =GaussianNB()
X_train,X_test,y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=10) #what is random_state?
nbc.fit(X_train,y_train)
y_pred = nbc.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))# what is does?
nbc.score(X_test,y_test) #?
nbc.score(X_test,y_test) #?


