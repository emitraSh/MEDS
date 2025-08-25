import numpy as np
import pandas as pd
import pickle
import joblib

from sklearn.model_selection import train_test_split,cross_val_score
import sklearn.ensemble
import sklearn.svm #support vector machines:
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.impute import KNNImputer

model_train_df = pd.read_csv('train.csv')
competition_test = pd.read_csv('test.csv')
bank_full_df = pd.read_csv('bank-full.csv')

model_train_df = model_train_df.drop('id', axis=1)
model_train_df = model_train_df.drop('day', axis=1)
model_train_df = model_train_df.drop('month', axis=1)
bank_full_df = bank_full_df.drop('day', axis=1)
bank_full_df = bank_full_df.drop('month', axis=1)
test_id = competition_test["id"]
competition_test = competition_test.drop('id', axis=1)
competition_test = competition_test.drop('day', axis = 1)
competition_test = competition_test.drop('month', axis = 1)

bank_full_df = bank_full_df[model_train_df.columns] # to arrange the order based on train
train_df = pd.concat([model_train_df, bank_full_df], ignore_index=True)

cols_to_check = ["job", "education", "contact", "poutcome"]
train_df = train_df[(train_df[cols_to_check].eq("unknown").sum(axis=1) <= 3)]

def set_data_type(df):
    df['job']= df['job'].map({
                              'unknown': np.nan,
                              'unemployed': 1,
                              'student': 2,
                              'housemaid':3,
                              'services':8,
                              'blue-collar': 9,
                              'technician': 10,
                              'retired': 7,
                              'admin.':8,
                              'management': 13,
                              'entrepreneur': 20})

    df['job']= df['job'].astype(float)
    df['marital'] = df['marital'].map({'married': 0,'single':1,'divorced':2})
    df['education'] = df['education'].map({'unknown':np.nan, 'primary':1, 'secondary':2, 'tertiary':3})
    df['education']= df['education'].astype(float)
    df['contact'] = df['contact'].map({'unknown':0,'cellular':1, 'telephone':2})
    df['poutcome'] = df['poutcome'].map({'unknown':0,'failure': 1,'other':2,'success':3})
    df['poutcome']= df['poutcome'].astype(float)
    df['default'] = df['default'].map({'yes':1,'no':0,0:0, 1:1}) #has credit in default
    df['housing'] = df['housing'].map({'yes':1,'no':0,0:0, 1:1})
    df['loan'] = df['loan'].map({'yes':1,'no':0,0:0, 1:1})
    if df.shape[1] == 15:
        df['y'] = df['y'].map({'yes':1,'no':0, 0:0, 1:1})
        df['y'] = df['y'].astype(int)
    return df
print(f"{train_df.shape}____{competition_test.shape}_______________________________________________")
train_df = set_data_type(train_df)
test_df = set_data_type(competition_test)

x_train = train_df.drop(columns=["y"])
y_train = train_df['y']

max_depth= 18
n_estimators= 120
min_samples_leaf= 3
criterion= "log_loss"

clf = sklearn.ensemble.RandomForestClassifier(max_depth= max_depth,
                                              n_estimators= n_estimators,
                                              min_samples_leaf= min_samples_leaf,
                                              criterion= criterion,random_state=42)
clf.fit(x_train, y_train)

y_proba = clf.predict_proba(competition_test)[:, 1]

submission = pd.DataFrame({
    "id": test_id,
    "y": y_proba
})

# 7. Save to CSV
submission.to_csv("submission.csv", index=False)
print(" submission.csv file created")





