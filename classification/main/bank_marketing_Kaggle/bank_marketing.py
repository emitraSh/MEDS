import numpy as np
import pandas as pd
import pickle
import joblib

#Visualization
import matplotlib.pyplot as plt
import seaborn as sns

#model
from sklearn.model_selection import train_test_split,cross_val_score
import optuna
import sklearn.ensemble
import sklearn.svm #support vector machines:
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.impute import KNNImputer


model_train_1_df = pd.read_csv('train.csv')
competition_test = pd.read_csv('test.csv')
bank_full_df = pd.read_csv('bank-full.csv')


#no use!
print(model_train_1_df.shape , bank_full_df.shape)
#print(model_train_df.head())
#print(model_train_df.describe())

model_train_1_df = model_train_1_df.drop('id', axis=1)
model_train_1_df = model_train_1_df.drop('day', axis=1)
model_train_1_df = model_train_1_df.drop('month', axis=1)
bank_full_df = bank_full_df.drop('day', axis=1)
bank_full_df = bank_full_df.drop('month', axis=1)

#merging the datasets________________________________________________________________________
bank_full_df = bank_full_df[model_train_1_df.columns] # to arrange the order based on train
model_train_df = pd.concat([model_train_1_df, bank_full_df], ignore_index=True)
#print(model_train_df.dtypes)
#____________________________________________________________________________________________

#eliminating null rows_______________________________________________________________________
cols_to_check = ["job", "education", "contact", "poutcome"]
#print(f"before_____________{model_train_df.shape}__________")
model_train_df = model_train_df[(model_train_df[cols_to_check].eq("unknown").sum(axis=1) <= 3)]
#print(f"after_____________{model_train_df.shape}__________")
#____________________________________________________________________________________________


"""print("----------- Data Types: ------------")
for column in model_train_df.columns:
   print(f"unique value of {column} : {model_train_df[column].unique()}")"""

"""print("--------- Null value ----------")
missing_values = model_train_df.isnull().sum()
missing_values = missing_values[missing_values != 0]
print(missing_values)"""

'''

Questions:
1. Even though there are no null values, but some categorical columns have unknown values, do I need to define those first?How?
2. Do I need to change the data types in code or its not necessary?

'''

#____________________________________________ setting data types __________________________________________
model_train_df['job']= model_train_df['job'].map({
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

model_train_df['job']= model_train_df['job'].astype(float)
model_train_df['marital'] = model_train_df['marital'].map({'married': 0,'single':1,'divorced':2})
model_train_df['education'] = model_train_df['education'].map({'unknown':np.nan, 'primary':1, 'secondary':2, 'tertiary':3})
model_train_df['education']= model_train_df['education'].astype(float)
model_train_df['contact'] = model_train_df['contact'].map({'unknown':0,'cellular':1, 'telephone':2})
model_train_df['poutcome'] = model_train_df['poutcome'].map({'unknown':0,'failure': 1,'other':2,'success':3})
model_train_df['poutcome']= model_train_df['poutcome'].astype(float)
model_train_df['default'] = model_train_df['default'].map({'yes':1,'no':0,0:0, 1:1}) #has credit in default
model_train_df['housing'] = model_train_df['housing'].map({'yes':1,'no':0,0:0, 1:1})
model_train_df['loan'] = model_train_df['loan'].map({'yes':1,'no':0,0:0, 1:1})
model_train_df['y'] = model_train_df['y'].map({'yes':1,'no':0, 0:0, 1:1})
model_train_df['y'] = model_train_df['y'].astype(int)



X = model_train_df.drop(columns=["y"])
Y = model_train_df['y']
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=10)
imputer = KNNImputer(n_neighbors=5,missing_values=np.nan)
x_train = pd.DataFrame(imputer.fit_transform(x_train))
x_test = pd.DataFrame(imputer.transform(x_test))
joblib.dump(imputer, 'x_train_imputed.pkl')
#imputer =joblib.load('x_train_imputed.pkl') #fitted function will be saved for future estimation


#finding optimal randomForest parameters with ____________________  OPTUNA  ____________________
"""parameters ={
    "max_depth" : 10,     #number of splits
    "n_estimators" : 100, #number of trees in random forest
    "min_samples_leaf" : 20,
    "criterion" : "gini",
    "max_features" : "log2",
    "random_state" : 42
}

def objective(trial, parameters):
    parameters["max_depth"] = trial.suggest_int("max_depth",2,20,log = True)
    parameters["n_estimators"] = trial.suggest_int("n_estimators",50,150,log = True)
    parameters["min_samples_leaf"] = trial.suggest_int("min_samples_leaf",2,32,log = True)
    parameters["criterion"] = trial.suggest_categorical("criterion", ['gini','entropy','log_loss'])

    return model(parameters)


def model(parameters):
    rf_model = sklearn.ensemble.RandomForestClassifier(
        max_depth = parameters["max_depth"],
        n_estimators = parameters["n_estimators"],
        min_samples_leaf = parameters["min_samples_leaf"],
        criterion = parameters["criterion"],
        random_state = parameters["random_state"])


         #It trains and tests your model multiple times on different “folds” (splits)
         #of the dataset and returns the performance scores for each run.
     
         #Why do we use it?
         #A single train/test split might give a biased estimate (lucky or unlucky split).
         #Cross-validation averages across several splits → more reliable estimate of model performance.
    
    

    scores = cross_val_score(rf_model, x_train, y_train, cv=3,scoring="roc_auc_ovr")
    return scores.mean()


func = lambda trial: objective(trial, parameters)
study = optuna.create_study(direction="maximize")
study.optimize(func, n_trials=50)
print(study.best_params)

print("Best trial:")
trial = study.best_trial
print(f" ROC AUC: {trial.value}")
print(" Best hyperparameters:")

for key, value in trial.params.items():
    print(f" {key}: {value}")

def save_model(parameters,filename= 'Name'):
    with open(filename, 'wb') as file:
        pickle.dump(parameters, file)

save_model(trial.params)"""

#___ OPTUNA RESULT :
#Best trial:
#ROC AUC: 0.9519410716849173
max_depth= 18
n_estimators= 120
min_samples_leaf= 3
criterion= "log_loss"
# Train final model with best parameters
print(f"{y_train.shape}")
best_clf = sklearn.ensemble.RandomForestClassifier(max_depth= max_depth,
                                                    n_estimators= n_estimators,
                                                    min_samples_leaf= min_samples_leaf,
                                                    criterion= criterion,random_state=42)
best_clf.fit(x_train, y_train)

#____________________________ proximity matrix  _________________________________

"""def proximity_impute_dataframe(df, clf, feature_cols, missing_markers):
    df_copy = df.copy()
    leaf_train = clf.apply(df_copy[feature_cols])  # apply function checks every sample by every tree and returns : (n_samples, n_trees)

    for row_idx, row in df_copy.iterrows():# row index is important because at the end we want to replace the data
        for feature, marker in missing_markers.items():
            print(f"which feature: {feature}, which row: {row_idx}")
            if row[feature] == marker:
                row_df = row[feature_cols].to_frame().T
                #leaf_unknown = clf.apply(row_df)  # (1, n_trees)
                leaf_unknown = clf.apply([row[feature_cols]])
                valid_idx = df_copy[feature] != marker
                valid_positions = np.where(valid_idx.to_numpy())[0]
                sample_idx = np.random.choice(valid_positions, size=5000, replace=False) # that makes the program lighter
                prox = np.mean(leaf_train[sample_idx] == leaf_unknown, axis=1)  # this line calculates mean  across all trees for each sample and returns an array that shows how similar each sample is to the mentioned row
                print(f"prox {prox}")

                # Weighted average for imputation
                estimate = np.average(
                    df_copy.loc[sample_idx, feature],
                    weights=prox
                )

                # Replace missing value
                df_copy.at[row_idx, feature] = estimate

    return df_copy

feature_cols = x_train.columns.tolist()
prox_x_train = proximity_impute_dataframe(
    x_train,
    best_clf,
    feature_cols,
    missing_markers={'job': -1,'education':0 }
)

print("Imputation finished.")
print(type(prox_x_train))
print(prox_x_train.head())

best_clf.fit(prox_x_train, y_train)"""
#________________________________________________________________________________
y_proba = best_clf.predict_proba(x_test)

print(f"{y_proba.shape}_______________")
if y_test.ndim > 1 and y_test.shape[1] > 1:
    y_test_fixed = np.argmax(y_test, axis=1)
else:
    y_test_fixed = y_test

# Compute ROC AUC on test set
print("Final Test ROC AUC:", roc_auc_score(y_test_fixed, y_proba[:,1], multi_class="ovr"))
#Final Test ROC AUC: 0.9518467627264986
#Final Test ROC AUC After using KNNimputer: 0.9517615488727839




