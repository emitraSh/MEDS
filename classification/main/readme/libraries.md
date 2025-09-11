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
      - model.feature_importances_ : it has a function, feature importance that shows which feature is most important
    - task parameters:
    - https://xgboost.readthedocs.io/en/stable/parameter.html?utm_source=chatgpt.com#parameters-for-tree-booster




### Questions:
- How can I use Optuna? install -> import as a library...
- n_jobs: -1 uses all CPU cores for speed? does it have to change depending on laptop hardware?which parameters have that dependence?
- random_state=42 makes the split reproducible/ does it mean duplicated are allowed?


# AutoGluon
AutoGluon is an open-source AutoML (Automated Machine Learning) toolkit developed by Amazon Web Services (AWS).

AutoGluon uses a multi-layered ensemble framework combined with automated hyperparameter tuning and neural architecture search (NAS) in some domains.

✅ Support for Multiple ML Tasks
- Tabular data (classification, regression)
- Image classification and object detection
- Text classification, summarization, etc.
- Multimodal tasks (combining text, images, etc.)

✅ Auto-Ensembling
- Automatically trains multiple models (e.g., XGBoost, LightGBM, neural nets) and combines them using stacking or bagging to improve performance.

✅ Hyperparameter Optimization

Uses Bayesian optimization and other methods under the hood to search for the best models and configurations.

```python
from autogluon.tabular import TabularPredictor

predictor = TabularPredictor(label="target_column").fit(train_data)
predictions = predictor.predict(test_data)
```
For Tabular Data (autogluon.tabular), it can include:
- Tree-Based Models
    - LightGBM (Gradient Boosted Decision Trees)
    - XGBoost
    - CatBoost
    - Random Forest
    - ExtraTrees

  - Linear Models
  - Logistic Regression
    - Linear Regression

- Neural Networks
    - Feedforward neural networks (MLP) using MXNet or PyTorch backend

- k-Nearest Neighbors (KNN)
  - Ensembles / Meta-Models
  - Bagging (e.g., multiple LightGBMs)
  - Stacking (Layer-wise model stacking — more on this below)

- Specialized Models
    - TabTransformer (for categorical variables via deep learning)

