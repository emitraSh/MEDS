## Encoding cyclical features using sine and cosine transformations.

- what cyclical feature means?
  - it starts over after a number of events, month: every 12 month, day: every 31 days
  - If you encode: January = 1, February = 2, ..., December = 12,Then December (12) and January (1) appear to be far apart numerically, when in fact they are next to each other in time. This misleads many machine learning models (especially linear models or tree-based models).
  - The idea is to represent each cyclical variable as a point on a circle.
```python
import numpy as np
df = []
df["day_sin"] = np.sin(2 * np.pi * df["day"] / 31)
df["day_cos"] = np.cos(2 * np.pi * df["day"] / 31)
```

📈 Why is this useful for ML models?

Machine learning algorithms (especially linear regression, logistic regression, decision trees, etc.) often do not handle cyclical relationships natively. Using sin and cos encodings helps the model "see" that:

January is close to December

Day 1 is close to Day 31

Without this, the model might treat them as being very far apart numerically.

## Applying a logarithm transformation to a feature

```python
df_copy['balance_log'] = np.sign(df_copy['balance']) * np.log1p(np.abs(df_copy['balance']))
```
Description of code:
- np.abs(df_copy['balance']): Takes the absolute value of the balance (because log of a negative number is undefined).

- np.log1p(x): Computes log(1 + x). This is numerically more stable for small values than log(x), and avoids issues when x = 0.

- np.sign(df_copy['balance']): Preserves the original sign of the balance (positive or negative).

- Multiplying by the sign re-applies the original sign after the log transform.

Why???
- Handling Skewed Distributions
  - Log transformation compresses large values and stretches small ones, making the data more normally distributed.
- Improving Model Performance
  - Many machine learning algorithms (like linear regression, logistic regression) perform better when the data is close to normal.
  - Log transformation can help linearize relationships between features and the target.
- Reducing the Impact of Outliers
  - Logarithmic scaling reduces the impact of large outliers by squashing large values.
- Better Visualization

🔎 How to Know When to Apply Log

- Histograms → Are they skewed right?

- Box plots → Are there outliers pulling the mean?

- Scatter plots with target → Does the relationship with the target variable look exponential or multiplicative?

- Skewness metric → Values >1 or <−1 are considered highly skewed.

## group-wise operation

This is useful for creating a new feature based on group-level statistics.
in another word combining a categorical feature with a related numerical feature.
For instance grouping jobs and assigning the mean value of the balance to each job and creating a new feature to hold the assigned value.

```python
df['avg_balance_by_job'] = df.groupby('job')['balance'].transform('mean')
df['median_duration_by_education'] = df.groupby('education')['duration'].transform('median')
```
## Interactions
One can combine different features for example education + job, and create new feature that holds related values.
or for example balance/ age

## signal leakage:
keeping data belonging to same groups either to the test/train df ex:
```python
# We'll do a "grouped" split to keep all of an artist's songs in one
# split or the other. This is to help prevent signal leakage.
def group_split(X, y, group, train_size=0.75):
    splitter = GroupShuffleSplit(train_size=train_size)
    train, test = next(splitter.split(X, y, groups=group))
    return (X.iloc[train], X.iloc[test], y.iloc[train], y.iloc[test])
```