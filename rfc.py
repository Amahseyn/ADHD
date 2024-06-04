from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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

# Find relevant features using tsfresh

# Ensure dataX contains only numeric data

# Scale the features
scaler = StandardScaler()
dataX = scaler.fit_transform(dataX)

# Split the data
x_train, x_test, y_train, y_test = train_test_split(
    dataX,
    dataY,
    test_size=.25,
    random_state=0,
    stratify=dataY)

# Train the RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators=10000, random_state=0)
rf_classifier.fit(x_train, y_train)

# Predict the test set results
y_pred = rf_classifier.predict(x_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)
