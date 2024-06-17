from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import RFE
from imblearn.over_sampling import SMOTE

import numpy as np
import pandas as pd

dataX = pd.read_csv("features.csv", sep=";").sort_values(by="ID")
dataY = pd.read_csv("patient_info.csv", sep=";").sort_values(by="ID")

# Fill missing values
dataX = dataX.fillna(0)

# Match X and Y data
dataY = dataY[dataY["ID"].isin(dataX["ID"])]
dataX = dataX[dataX["ID"].isin(dataY["ID"])]

dataY = dataY.set_index("ID")
dataX = dataX.set_index("ID")

dataY = dataY["ADHD"].copy()

# Scale the features
scaler = StandardScaler()
dataX = scaler.fit_transform(dataX)

# Feature Selection with RFE
rf_selector = RandomForestClassifier(n_estimators=100, random_state=0)
selector = RFE(rf_selector, n_features_to_select=10, step=1)
dataX = selector.fit_transform(dataX, dataY)

# Split the data
x_train, x_test, y_train, y_test = train_test_split(
    dataX,
    dataY,
    test_size=0.25,
    random_state=0,
    stratify=dataY)

# Handle class imbalance using SMOTE
sm = SMOTE(random_state=0)
x_train, y_train = sm.fit_resample(x_train, y_train)

# Define classifiers for ensemble
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=0)
log_classifier = LogisticRegression(random_state=0)
svc_classifier = SVC(probability=True, random_state=0)

# Create a voting classifier
voting_classifier = VotingClassifier(estimators=[
    ('rf', rf_classifier),
    ('log', log_classifier),
    ('svc', svc_classifier)
], voting='soft')

# Define a grid of hyperparameters for grid search
param_grid = {
    'rf__n_estimators': [50, 100, 200],
    'rf__max_depth': [None, 10, 20, 30],
    'rf__min_samples_split': [2, 5, 10],
    'rf__min_samples_leaf': [1, 2, 4],
    'log__C': [0.1, 1, 10],
    'svc__C': [0.1, 1, 10],
    'svc__kernel': ['linear', 'rbf']
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(estimator=voting_classifier, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(x_train, y_train)

# Get the best parameters from grid search
best_params = grid_search.best_params_

# Train the VotingClassifier with the best parameters
best_voting_classifier = VotingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(**{k.split('__')[1]: v for k, v in best_params.items() if k.startswith('rf')}, random_state=0)),
        ('log', LogisticRegression(**{k.split('__')[1]: v for k, v in best_params.items() if k.startswith('log')}, random_state=0)),
        ('svc', SVC(**{k.split('__')[1]: v for k, v in best_params.items() if k.startswith('svc')}, probability=True, random_state=0))
    ],
    voting='soft'
)
best_voting_classifier.fit(x_train, y_train)

# Predict the test set results
y_pred = best_voting_classifier.predict(x_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)
print(f"Accuracy: {accuracy}")

# Perform cross-validation to check for overfitting
cross_val_scores = cross_val_score(best_voting_classifier, dataX, dataY, cv=5)
print(f"Cross-Validation Scores: {cross_val_scores}")
print(f"Mean Cross-Validation Score: {cross_val_scores.mean()}")
