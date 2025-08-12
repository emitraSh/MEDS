import pprint

import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from logesticRegression import LogisticRegression
import pickle

df = pd.read_csv('breast-cancer.csv')

#print(df.head())
#print(df.describe())

#hist_diagnosis =px.histogram(data_frame=df, x='diagnosis', color='diagnosis', color_discrete_sequence=px.colors.sequential.Reds)
#hist_diagnosis.show()

#hist_size_diagnosis = px.histogram(df,x='area_mean', color='diagnosis', color_discrete_sequence=px.colors.sequential.Reds)
#hist_size_diagnosis.show()

df.drop('id', axis=1, inplace=True)

df['diagnosis'] = (df['diagnosis'] == 'M').astype(int) # M->1 , B ->0

corr = df.corr() # creates correlation matrix - default is pearson correlation
#plt.figure(figsize = (20,20))
#sns.heatmap(corr,annot=True)
#plt.show()

cor_diagnosis = abs(corr['diagnosis'])
#print(f"______________-------{dict(cor_diagnosis)}-------_____________---")
feature_selection = cor_diagnosis[cor_diagnosis > 0.2]
feature_names = [index for index, value in feature_selection.items()] # it returns a list of feature names that have correlation above 0.2 with diagnosis
feature_names.remove('diagnosis') # it is redundant
#pprint.pprint(feature_names)

X = df[feature_names]
y = df['diagnosis']

X_train,X_test,y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=10)



def standardization(X):
    numeric_X = X.select_dtypes(include=np.number)
    mean = numeric_X.mean()
    std = numeric_X.std()
    standard_X = (numeric_X - mean) / std
    return standard_X

X_train_stndrd = standardization(X_train)
X_test_stndrd = standardization(X_test)

lg = LogisticRegression()
#lg.fit(X_train_stndrd, y_train, 100000)

#lg.save_model("logistic_model.pkl")
#pklfile = pd.read_pickle("logistic_model.pkl")
#print(pklfile)

with open("model.pkl", "rb") as f:
    my_data = pickle.load(f)

print(type(my_data))


lg.predict(X_test_stndrd, my_data['W'], my_data['b'])


