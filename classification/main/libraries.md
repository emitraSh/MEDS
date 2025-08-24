# sklearn:
- sklearn.preprocessing 
  - LabelEncoder : it converts categories in string to integers
    - fit_transform(datasetName): learnst the mapping of each unique category and applies the mapping to ds.
- sklearn.model_selection
  - StratifiedKFold: it creates a cross validation object of K folds.
    - n_splits: default 5, number of folds
    - shuffle: bool, weather to shuffel samples of each class before dividing to batches
    - random_state: if shuffle = True the random state gets an integer as a seed to maintain the order within different runs
    - https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html
- sklearn.metrics
  - roc_auc_score : it defines the accuracy of your model, for binary and categorical classification.
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
# xgboost : gradient-boosted decision trees
- parameters:
    - Optuna : is an automatic hyperparameter optimization software framework. it is used to determine parameters.
    - general parameters: which booster is to be used? ex:tree or linear? default is "gbtree"
    - <u>booster parameters</u>:
    - some booster parameters are dependent on hardware:
      - tree_method
      - n_jobs: Depends on CPU cores -> -1 uses all cores, lower numbers on laptop is more optimal
      - n_estimators: Affects training time, not accuracy: More trees → more time and memory usage. If your laptop is slow, you might reduce this and rely on early_stopping_rounds.
      - max_bin:Higher bins = more memory and training time. On low-RAM laptops, keep it modest.
      - gpu_id: ou might need to choose which GPU to run on; not relevant if you have only one GPU.
    - task parameters:
    - https://xgboost.readthedocs.io/en/stable/parameter.html?utm_source=chatgpt.com#parameters-for-tree-booster




### Questions:
- How can I use Optuna? install -> import as a library...
- n_jobs: -1 uses all CPU cores for speed? does it have to change depending on laptop hardware?which parameters have that dependence?
- random_state=42 makes the split reproducible/ does it mean duplicated are allowed?