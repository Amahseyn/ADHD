import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the data
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

# Ensure dataX contains only numeric data
# Assuming dataX is all numeric, otherwise use appropriate encoding methods for categorical data

# Scale the features
scaler = StandardScaler()
dataX = scaler.fit_transform(dataX)
print(len(dataX))
# Reshape data for Conv1D layer
dataX = dataX.reshape((dataX.shape[0], dataX.shape[1], 1))

# Split the data
x_train, x_test, y_train, y_test = train_test_split(
    dataX,
    dataY,
    test_size=0.25,
    random_state=0,
    stratify=dataY)
# Build the convolutional neural network model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(filters=16, kernel_size=3, activation='elu', input_shape=(x_train.shape[1], 1)),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='elu'),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(8, activation='elu'),
    

    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(x_train, y_train, epochs=150, batch_size=4, validation_split=0.5)

# Predict the test set results
y_pred_prob = model.predict(x_test)
y_pred = (y_pred_prob > 0.5).astype(int)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)