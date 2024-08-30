import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt


data = pd.read_csv('data_set.csv') 


X = data[['Ir', 'T']].values
y = data['Power'].values 


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# L2 regularization
   
model = keras.Sequential([
    keras.layers.Dense(512, activation='relu', input_shape=(X.shape[1],),
                      5kernel_regularizer=keras.regularizers.l2(0.01)),
    keras.layers.Dense(1024, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
    keras.layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
    keras.layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
    keras.layers.Dense(1, activation='linear')
])


# mean squared error loss
model.compile(optimizer='adam', loss='mean_squared_error')


# Train the model
history = model.fit(X_train, y_train, epochs=200, batch_size=64, verbose=0, validation_data=(X_test, y_test))


# Make predictions on the test set
y_pred = model.predict(X_test)


# metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


# Print the metrics
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared (R^2) Score: {r2}")




# Plot Predicted vs. Actual Output
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Output")
plt.ylabel("Predicted Output")
plt.title("Predicted vs. Actual Output")
plt.grid()
plt.show()


# Plot Loss History
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss History')
plt.grid()
plt.show()
