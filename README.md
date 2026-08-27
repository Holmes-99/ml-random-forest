# Random Forest Classifier with Tuning and Feature Importance

A random forest classifier trained on the Breast Cancer Wisconsin dataset, tuned with grid search and inspected for feature importance.

## Dataset
The Breast Cancer Wisconsin dataset from scikit-learn (`load_breast_cancer`). 30 numeric measurements per sample, target is malignant or benign.

## What the notebook does
- Loads and splits the data, stratified on the target
- Tunes `n_estimators` and `max_depth` with `GridSearchCV` (5-fold cross-validation, scored by F1)
- Evaluates the best model with cross-validation plus test accuracy, precision, recall, and F1
- Computes and plots the top 10 feature importances as a horizontal bar chart

## Key ideas
- A random forest is an ensemble of many decision trees, each trained on a random subset of data and features, that vote on the final prediction. Averaging many differently-biased trees generalizes better than relying on a single tree.
- Grid search tries every combination of hyperparameter values with cross-validation, which gives a more reliable comparison than judging performance from a single train/test split.
- Feature importance shows how much each feature contributes to the forest's splits on average, which points to what the model is actually relying on to make predictions.

## Tools
Python, scikit-learn, matplotlib

## How to run
```
pip install pandas scikit-learn matplotlib numpy
jupyter notebook random_forest.ipynb
```
Run the cells top to bottom. Grid search over 9 parameter combinations with 5-fold cross-validation takes a few seconds to a couple of minutes depending on the machine.
