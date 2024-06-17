import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,normalize
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the data
dataX = pd.read_csv("features.csv", sep=";").sort_values(by="ID")
dataY = pd.read_csv("patient_info.csv", sep=";").sort_values(by="ID")

# Fill missing values with appropriate methods
for column in dataX.columns:
    if dataX[column].dtype in [np.float64, np.int64]:
        if dataX[column].isnull().all():
            # If the entire column is NaN, drop the column
            dataX.drop(columns=[column], inplace=True)
        else:
            # For numeric columns, use median imputation
            dataX[column].fillna(dataX[column].median(), inplace=True)
    else:
        if dataX[column].isnull().all():
            # If the entire column is NaN, drop the column
            dataX.drop(columns=[column], inplace=True)
        else:
            # For categorical columns, use mode (most frequent) imputation
            dataX[column].fillna(dataX[column].mode()[0], inplace=True)

# Check for any remaining NaNs
assert not dataX.isnull().values.any(), "There are still missing values in the dataset."

# Match X and Y data
dataX = dataX.set_index("ID")
dataY = dataY.set_index("ID")
dataY = dataY[dataY.index.isin(dataX.index)]
dataX = dataX[dataX.index.isin(dataY.index)]

dataY = dataY["ADHD"].copy()

# Scale the features
scaler = StandardScaler()
dataX = scaler.fit_transform(dataX)
print(dataX.shape)
print(dataY.shape)
# Apply PCA
pca = PCA(n_components=0.7)  # Adjust the number of components as needed
dataX = pca.fit_transform(dataX)

# Split the data before augmentation
x_train, x_test, y_train, y_test = train_test_split(
    dataX,
    dataY,
    test_size=0.25,
)

def add_noise(data, noise_factor=0.15):
    noise = np.random.randn(*data.shape) * noise_factor
    augmented_data = data + noise
    return augmented_data

# Augment the training data with noise
x_train_noisy = add_noise(x_train)

print(f"Original training data length: {len(x_train)}")
print(f"Noisy training data length: {len(x_train_noisy)}")

# Reshape data for Conv1D layer
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
x_train_noisy = x_train_noisy.reshape((x_train_noisy.shape[0], x_train_noisy.shape[1], 1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

# Build the convolutional neural network model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(filters=32, kernel_size=2, activation='relu', input_shape=(x_train.shape[1], 1)),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Conv1D(filters=16, kernel_size=2, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),

    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=200, restore_best_weights=True)

# Train the model using original and noisy data separately
history_noisy = model.fit(x_train, y_train, epochs=200, batch_size=8, validation_split=0.25, callbacks=[early_stopping])

# Predict the test set results
y_pred_prob = model.predict(x_test)
y_pred = (y_pred_prob > 0.5).astype(int)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)
print(f"Accuracy: {accuracy}")
