"""
def proximity_to_training(rf, x_train, x_new):
    n_trees = n_estimators
    leaf_train = best_clf.apply(x_train)  # (n_samples, n_trees)
    leaf_new = best_clf.apply([x_new])[0]  # (n_trees,)

    proximities = np.mean(leaf_train == leaf_new, axis=1)
    return proximities  # vector of length n_samples


row_idx =0
for x_unknown in x_train:
    if x_unknown[1] == 0 or x_unknown[-1] == -1:

        prox = proximity_to_training(best_clf, x_train, x_unknown)
        if x_unknown[1] == 0:
            missing_feature_index = 1
            #estimate = np.average(X[:, missing_feature_index], weights=prox)
            missing_feature = 'job'
            x_train[row_idx,'job'] = np.nan
            estimate = np.average(x_train[missing_feature].dropna(), weights=prox[~x_train[missing_feature].isna()])
            print(f"{row_idx}=row_idx____  {missing_feature}=missing_feature____ {estimate}=estimate____")

            # Replace missing value
            x_train.loc[row_idx, missing_feature] = estimate

        if x_unknown[-1] == -1:
            missing_feature_index = -1
            missing_feature = 'poutcome'
            x_train[row_idx,missing_feature] = np.nan
            estimate = np.average(x_train[missing_feature].dropna(), weights=prox[~x_train[missing_feature].isna()])

            # Replace missing value
            x_train.loc[row_idx, missing_feature] = estimate
            print(f"{row_idx}=row_idx____  {missing_feature}=missing_feature____ {estimate}=estimate____")

    row_idx += 1

best_clf.fit(x_train, y_train)"""


def compute_proximity_vector(rf, leaf_train, x_new, feature_cols):
    # Debugging info
    print("DEBUG before conversion:", type(x_new))

    if isinstance(x_new, np.ndarray):
        # Convert ndarray -> DataFrame with feature names
        x_new = pd.DataFrame([x_new], columns=feature_cols)
    elif isinstance(x_new, pd.Series):
        # Convert Series -> DataFrame
        x_new = x_new.to_frame().T
        x_new = x_new[feature_cols]  # enforce column order
    elif isinstance(x_new, pd.DataFrame):
        # Ensure column order
        x_new = x_new[feature_cols]
    else:
        raise ValueError(f"Unexpected type for x_new: {type(x_new)}")

    print("DEBUG after conversion:", type(x_new), x_new.shape)

    leaf_new = rf.apply(x_new)[0]  # should now be clean
    return np.mean(leaf_train == leaf_new, axis=1)


"""def proximity_impute_dataframe(df, best_clf, feature_cols, missing_markers):

    # Prepare numeric training matrix that matches the RF training order
    df_copy = df.copy()
    X_train_np = df[feature_cols].to_numpy()   # shape (n_train, n_features)
    leaf_train = best_clf.apply(X_train_np)    # shape (n_train, n_trees)

    n_train = X_train_np.shape[0]

    # iterate rows with index
    for row_idx in df.index:
        row = df.loc[row_idx, feature_cols]   # pandas Series

        # detect which columns for this row are "missing" according to markers
        missing_cols = []
        for col, marker in missing_markers.items():
            val = row[col]
            # missing if NaN or equals marker
            if pd.isna(val) or (marker is not None and val == marker):
                missing_cols.append(col)

        if not missing_cols:
            continue  # nothing to impute for this row

        # build x_unknown: a numeric vector for RF.apply (use placeholder for missing cols only)
        x_unknown = row.to_numpy().astype(float).copy()

        # placeholder strategy: column mean (from training values) ignoring NaNs
        for col in missing_cols:
            idx = feature_cols.index(col)
            col_vals = X_train_np[:, idx]
            placeholder = np.nanmean(col_vals)   # safe even if some NaNs
            x_unknown[idx] = placeholder

        # compute proximity weights to all training samples
        prox = compute_proximity_vector(leaf_train, best_clf, x_unknown)  # shape (n_train,)

        # for each missing column, compute estimate and write back
        for col in missing_cols:
            idx = feature_cols.index(col)

            # mask to select training rows that have a valid value for 'col'
            mask = ~np.isnan(X_train_np[:, idx])
            # if there is a special marker in training set, exclude those too
            marker = missing_markers.get(col, None)
            if marker is not None:
                mask = mask & (X_train_np[:, idx] != marker)

            train_vals = X_train_np[mask, idx]
            weights = prox[mask]

            if train_vals.size == 0:
                # no training info, fallback to global mean / None
                est = np.nan
            else:
                if weights.sum() <= 0:
                    # no proximity mass -> fallback to simple mean
                    est = np.nanmean(train_vals)
                else:
                    # decide if column should be treated as categorical:
                    # if df[col] is non-numeric type (object/category) treat as categorical
                    if not np.issubdtype(df[col].dtype, np.number):
                        # weighted mode (best category by summed weights)
                        unique_vals = np.unique(train_vals)
                        best_val = None
                        best_score = -1.0
                        for v in unique_vals:
                            score = weights[train_vals == v].sum()
                            if score > best_score:
                                best_score = score
                                best_val = v
                        est = best_val
                    else:
                        # numeric: weighted average
                        est = np.average(train_vals, weights=weights)

            # write the estimate back to dataframe (use loc)
            df.loc[row_idx, col] = est

    return pd.DataFrame(df, columns=feature_cols)"""


def proximity_impute_dataframe(df, clf, feature_cols, missing_markers):
    df_copy = df.copy()

    for row_idx, row in df_copy.iterrows():
        for feature, marker in missing_markers.items():
            if row[feature] == marker:
                # Compute proximity
                print("whats up")
                leaf_train = clf.apply(df_copy[feature_cols])  # shape (n_samples, n_trees)

                prox = compute_proximity_vector(
                    clf,
                    leaf_train,
                    row[feature_cols],
                    feature_cols
                )
                print("here")

                # Weighted average for imputation
                valid = df_copy[feature] != marker
                estimate = np.average(df_copy.loc[valid, feature], weights=prox[valid])

                # Replace
                df_copy.at[row_idx, feature] = estimate

    return df_copy


feature_cols = x_train.columns.tolist()
prox_x_train = proximity_impute_dataframe(x_train, best_clf, feature_cols, missing_markers={'poutcome': -1, 'job': 0})
print("is it finished?")
print(type(prox_x_train))
print(prox_x_train.head())
best_clf.fit(prox_x_train, y_train)


#__________________________________

def proximity_impute_dataframe(df, clf, feature_cols, missing_markers):
    df_copy = df.copy()
    df_copy_without_job = df_copy[df_copy['job'] != 0]

    # Compute leaf indices once for the whole training data
    leaf_train = clf.apply(df_copy[feature_cols])  # (n_samples, n_trees)

    for row_idx, row in df_copy.iterrows():
        for feature, marker in missing_markers.items():
            if row[feature] == marker:
                # Compute proximity vector for this row
                leaf_unknown = clf.apply([row[feature_cols]])  # (1, n_trees)
                prox = np.mean(leaf_train == leaf_unknown, axis=1)  # (n_samples,)

                # Weighted average for imputation
                valid = df_copy[feature] != marker
                estimate = np.average(
                    df_copy.loc[valid, feature],
                    weights=prox[valid]
                )

                # Replace missing value
                df_copy.at[row_idx, feature] = estimate

    return df_copy


# ==== usage ====
feature_cols = x_train.columns.tolist()
prox_x_train = proximity_impute_dataframe(
    x_train,
    best_clf,
    feature_cols,
    missing_markers={'job': 0}
)

print("Imputation finished.")
print(type(prox_x_train))
print(prox_x_train.head())

best_clf.fit(prox_x_train, y_train)