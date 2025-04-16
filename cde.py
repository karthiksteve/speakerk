# forecast_aep.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import Model, Sequential
from keras.layers import Input, Dense, LSTM, Dropout, RepeatVector
from keras.callbacks import EarlyStopping

# --------- Step 1: Load Data ----------
file_path = "C:\\Users\\speak\\Downloads\\archive\\AEP_hourly.csv"

# Check if file exists
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File not found: {file_path}")

# Load and check the data
data = pd.read_csv(file_path, parse_dates=['Datetime'])
data.set_index('Datetime', inplace=True)
print("Columns found:", data.columns)

# Use correct column name based on your dataset
column_name = 'AEP_MW'  # Adjust if needed
if column_name not in data.columns:
    raise KeyError(f"Column '{column_name}' not found in dataset.")

# --------- Step 2: Preprocessing ----------
data.ffill(inplace=True)

scaler = MinMaxScaler()
data['Scaled'] = scaler.fit_transform(data[[column_name]])

def create_sequences(data, time_steps=24):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

sequence_length = 24
X, y = create_sequences(data['Scaled'].values, sequence_length)
X = X.reshape((X.shape[0], X.shape[1], 1))

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# --------- Step 3: Autoencoder ----------
input_layer = Input(shape=(sequence_length, 1))
encoded = LSTM(64, activation='relu', return_sequences=False)(input_layer)
decoded = RepeatVector(sequence_length)(encoded)
decoded = LSTM(64, return_sequences=True)(decoded)
decoded = Dense(1)(decoded)

autoencoder = Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X_train, X_train, epochs=30, batch_size=64,
                validation_split=0.2, callbacks=[EarlyStopping(patience=5)], verbose=1)

# --------- Step 4: Encoder Output to LSTM ----------
encoder = Model(inputs=input_layer, outputs=encoded)
X_train_encoded = encoder.predict(X_train)
X_test_encoded = encoder.predict(X_test)

X_train_encoded = X_train_encoded.reshape((X_train_encoded.shape[0], 1, X_train_encoded.shape[1]))
X_test_encoded = X_test_encoded.reshape((X_test_encoded.shape[0], 1, X_test_encoded.shape[1]))

# --------- Step 5: LSTM Model ----------
lstm_model = Sequential()
lstm_model.add(LSTM(50, activation='relu', input_shape=(X_train_encoded.shape[1], X_train_encoded.shape[2])))
lstm_model.add(Dropout(0.2))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer='adam', loss='mse')
lstm_model.fit(X_train_encoded, y_train, epochs=30, batch_size=64,
               validation_split=0.2, callbacks=[EarlyStopping(patience=5)], verbose=1)

# --------- Step 6: Evaluate Model ----------
y_pred = lstm_model.predict(X_test_encoded)

y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
y_pred_inv = scaler.inverse_transform(y_pred).flatten()

mse = mean_squared_error(y_test_inv, y_pred_inv)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_inv, y_pred_inv)
mape = np.mean(np.abs((y_test_inv - y_pred_inv) / y_test_inv)) * 100

print("\n--- Model Evaluation ---")
print(f"MSE   : {mse:.2f}")
print(f"RMSE  : {rmse:.2f}")
print(f"MAE   : {mae:.2f}")
print(f"MAPE  : {mape:.2f}%")

# --------- Step 7: Plot Results ----------
plt.figure(figsize=(14, 6))
plt.plot(y_test_inv[:100], label='Actual', color='blue')
plt.plot(y_pred_inv[:100], label='Predicted', color='red', linestyle='--')
plt.title("AEP Power Consumption Prediction (First 100 Points)")
plt.xlabel("Time Step")
plt.ylabel("Power Consumption (MW)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("forecast_plot.png", dpi=300)
plt.show()
