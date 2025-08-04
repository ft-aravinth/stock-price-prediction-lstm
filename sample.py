import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

class StockPricePredictor:
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def load_data(self, file_path):
        df = pd.read_csv(file_path, parse_dates=['Date'])
        df = df.sort_values('Date')
        df = df[['Close']]  # Use only 'Close' column
        df.dropna(inplace=True)
        self.data = df
        return df

    def prepare_data(self, sequence_length=60, test_split=0.2):
        scaled = self.scaler.fit_transform(self.data)
        X, y = [], []
        for i in range(sequence_length, len(scaled)):
            X.append(scaled[i-sequence_length:i])
            y.append(scaled[i])
        X, y = np.array(X), np.array(y)
        split = int(len(X) * (1 - test_split))
        self.X_train, self.X_test = X[:split], X[split:]
        self.y_train, self.y_test = y[:split], y[split:]

    def build_model(self):
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(self.X_train.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(50))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        self.model = model

    def train(self, epochs=30, batch_size=32):
        self.model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size, verbose=1)

    def evaluate(self):
        pred_scaled = self.model.predict(self.X_test)
        pred = self.scaler.inverse_transform(pred_scaled)
        actual = self.scaler.inverse_transform(self.y_test)
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        print(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}")
        self.plot(actual, pred)

    def plot(self, actual, predicted):
        plt.figure(figsize=(10, 5))
        plt.plot(actual, label="Actual")
        plt.plot(predicted, label="Predicted")
        plt.legend()
        plt.title("Stock Price Prediction")
        plt.show()


if __name__ == "__main__":
    predictor = StockPricePredictor()
    predictor.load_data("sample_stock.csv")
    predictor.prepare_data()
    predictor.build_model()
    predictor.train()
    predictor.evaluate()
