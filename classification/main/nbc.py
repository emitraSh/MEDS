import numpy as np
from collections import defaultdict
from scipy.stats import norm,dirichlet,beta

def featureTypes(col):
    unique_val = np.unique(col)
    if col.dtype.kind in {'i', 'f'}:  # with Numpy library checks if data type is integer or float
        if set(unique_val) <= {0, 1}:
            feature_type ='Binary'
        elif len(unique_val) <= 5:
            feature_type = 'categorical'
        else:
            feature_type = 'real'
    else:
        feature_type = 'categorical'
    return feature_type

#NBC _ MLE
def fit_NBC_MLE(X,y,feature_types):
    classes = np.unique(y)
    classes_prob = {}
    class_scores = {}


    #defining feature types
    if feature_types == [] or len(feature_types) != X.shape[1]:
        feature_types = []
        for i in range(X.shape[1]):
            col = X[:,i] #all values in column i - X[i,:]: all values in row i
            feature_types.append(featureTypes(col))

    for c in classes:
        X_c = X[y==c] #selects the rows that y equals to c
        classes_prob[c] = len(X_c) /len(X)
        log_prob = np.log(classes_prob[c])

        for i in X_c.shape[1]:
            ftype = feature_types[i]
            xic = X_c[:,i]
            if ftype == 'real':
                mu = np.mean(xic)
                sigma = np.std(xic) + 1e-6 #??
                for row in xic:
                    log_prob += norm.logpdf(row, loc=mu, scale=sigma)
            elif ftype == 'Binary':
                p = np.mean(xic)
                log_prob += sum(xic)*np.log(p)
            elif ftype == 'categorical':
                c_i = np.unique(xic)
                for cat in c_i:
                    c_i_row = xic[xic == cat]
                    log_prob += np.log(len(c_i_row)/len(c_i)) #???????????
        class_scores[c] = log_prob

    predicted_class = max(class_scores, key=class_scores.get)
    return predicted_class












