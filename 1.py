
import warnings

from sklearn import metrics
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
from aeon.classification.feature_based import Catch22Classifier, FreshPRINCEClassifier
from aeon.datasets import load_basic_motions, load_italy_power_demand
from aeon.registry import all_estimators
from aeon.transformations.collection.feature_based import Catch22
all_estimators("classifier", filter_tags={"algorithm_type": "feature"})
dataX = pd.read_csv("/home/mio/Documents/code/ADHD/preprocessed_data/activity_features.csv", sep=";").sort_values(by="ID")
dataY = pd.read_csv("patient_info.csv", sep=";").sort_values(by="ID")

# Fill missing values with appropriate methods
for column in dataX.columns:
    if dataX[column].dtype in [np.float64, np.int64]:
        if dataX[column].isnull().all():
            # If the entire column is NaN, replace with 0 or drop the column
            dataX[column].fillna(0, inplace=True)
        else:
            # For numeric columns, use median imputation
            dataX[column].fillna(dataX[column].median(), inplace=True)
    else:
        if dataX[column].isnull().all():
            # If the entire column is NaN, replace with most frequent or drop the column
            dataX[column].fillna(dataX[column].mode()[0], inplace=True)
        else:
            # For categorical columns, use mode (most frequent) imputation
            dataX[column].fillna(dataX[column].mode()[0], inplace=True)

# Check for any remaining NaNs


# Match X and Y data
dataY = dataY[dataY["ID"].isin(dataX["ID"])]
dataX = dataX[dataX["ID"].isin(dataY["ID"])]

dataY = dataY.set_index("ID")
dataX = dataX.set_index("ID")

dataY = dataY["ADHD"].copy()



# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    dataX,
    dataY,
    test_size=0.25,
    random_state=0,
    stratify=dataY)
c22 = Catch22()
x_trans = c22.fit_transform(X_train)
x_trans.shape
(67, 22)
c22cls = Catch22Classifier()
c22cls.fit(X_train, y_train)
c22_preds = c22cls.predict(X_test)
metrics.accuracy_score(y_test, c22_preds)
