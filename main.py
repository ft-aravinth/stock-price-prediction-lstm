# AI/ML Interview Project: Stock Price Prediction using LSTM
# Clean, well-documented code for technical interviews

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import warnings

warnings.filterwarnings('ignore')


class StockPricePredictor:
    """
    A complete ML pipeline for stock price prediction using LSTM neural networks.

    This class demonstrates:
    - Data preprocessing and feature engineering
    - Time series preparation for deep learning
    - LSTM model architecture
    - Model training and evaluation
    - Prediction and visualization
    """

    def __init__(self):
        self.data = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_sample_data(self):
        """
        Create sample stock data for demonstration.
        In real scenario, replace with CSV loading: pd.read_csv('your_file.csv')
        """
        print("📊 Creating sample stock data...")

        # Generate realistic stock price data
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')

        # Simulate stock price with trend and volatility
        initial_price = 100
        prices = [initial_price]

        for i in range(1, len(dates)):
            # Random walk with slight upward trend
            change = np.random.normal(0.001, 0.02)  # 0.1% daily growth, 2% volatility
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)

        # Create DataFrame
        self.data = pd.DataFrame({
            'Date': dates,
            'Close': prices
        })

        # Add some technical indicators
        self.data['SMA_10'] = self.data['Close'].rolling(window=10).mean()
        self.data['SMA_50'] = self.data['Close'].rolling(window=50).mean()
        self.data['Volatility'] = self.data['Close'].rolling(window=20).std()

        # Remove NaN values
        self.data = self.data.dropna()

        print(f"✅ Generated {len(self.data)} days of stock data")
        print(f"📈 Price range: ${self.data['Close'].min():.2f} - ${self.data['Close'].max():.2f}")

        return self.data

    def load_csv_data(self, file_path):
        """
        Load data from CSV file
        Expected columns: Date, Close (and optionally: Open, High, Low, Volume)
        """
        print(f"📂 Loading data from {file_path}...")

        try:
            self.data = pd.read_csv(file_path)

            # Standardize column names
            column_mapping = {
                'date': 'Date', 'DATE': 'Date',
                'close': 'Close', 'CLOSE': 'Close', 'Close Price': 'Close'
            }

            for old_name, new_name in column_mapping.items():
                if old_name in self.data.columns:
                    self.data.rename(columns={old_name: new_name}, inplace=True)

            # Convert date column
            if 'Date' in self.data.columns:
                self.data['Date'] = pd.to_datetime(self.data['Date'])
                self.data = self.data.sort_values('Date')

            print(f"✅ Loaded {len(self.data)} records")
            print(f"📊 Columns: {list(self.data.columns)}")

        except Exception as e:
            print(f"❌ Error loading file: {e}")
            print("🔄 Using sample data instead...")
            self.load_sample_data()

        return self.data

    def create_features(self):
        """
        Create technical indicators and features for better prediction
        """
        print("🔧 Creating technical features...")

        df = self.data.copy()

        # Moving averages (trend indicators)
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['EMA_12'] = df['Close'].ewm(span=12).mean()

        # Price momentum
        df['Returns'] = df['Close'].pct_change()
        df['Price_Change'] = df['Close'].diff()

        # Relative Strength Index (RSI)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # Volatility
        df['Volatility'] = df['Close'].rolling(window=10).std()

        # Price position (high-low position)
        if 'High' in df.columns and 'Low' in df.columns:
            df['Price_Position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])

        # Remove NaN values
        self.data = df.dropna()

        print(f"✅ Created features. Dataset shape: {self.data.shape}")
        return self.data

    def prepare_data(self, sequence_length=60, features=['Close'], test_split=0.2):
        """
        Prepare data for LSTM model

        Args:
            sequence_length: Number of previous days to use for prediction
            features: List of feature columns to use
            test_split: Fraction of data to use for testing
        """
        print(f"📋 Preparing data with sequence length: {sequence_length}")

        # Select features
        if isinstance(features, str):
            features = [features]

        # Ensure all features exist
        available_features = [f for f in features if f in self.data.columns]
        if not available_features:
            available_features = ['Close']
            print("⚠️ Using only 'Close' price as feature")

        # Prepare feature matrix
        feature_data = self.data[available_features].values

        # Scale the data
        scaled_data = self.scaler.fit_transform(feature_data)

        # Create sequences
        X, y = [], []

        for i in range(sequence_length, len(scaled_data)):
            # Use previous 'sequence_length' days as input
            X.append(scaled_data[i - sequence_length:i])
            # Predict next day's close price (first column)
            y.append(scaled_data[i, 0])

        X, y = np.array(X), np.array(y)

        # Split into train and test
        split_index = int(len(X) * (1 - test_split))

        self.X_train = X[:split_index]
        self.X_test = X[split_index:]
        self.y_train = y[:split_index]
        self.y_test = y[split_index:]

        print(f"📊 Data split:")
        print(f"   Training: {self.X_train.shape[0]} samples")
        print(f"   Testing:  {self.X_test.shape[0]} samples")
        print(f"   Features: {self.X_train.shape[2]} features")

        return self.X_train, self.X_test, self.y_train, self.y_test

    def build_model(self, lstm_units=[50, 50], dropout_rate=0.2):
        """
        Build LSTM neural network model

        Args:
            lstm_units: List of LSTM layer units
            dropout_rate: Dropout rate for regularization
        """
        print("🏗️ Building LSTM model...")

        self.model = Sequential()

        # First LSTM layer
        self.model.add(LSTM(
            units=lstm_units[0],
            return_sequences=True if len(lstm_units) > 1 else False,
            input_shape=(self.X_train.shape[1], self.X_train.shape[2])
        ))
        self.model.add(Dropout(dropout_rate))

        # Additional LSTM layers
        for i in range(1, len(lstm_units)):
            return_sequences = i < len(lstm_units) - 1
            self.model.add(LSTM(units=lstm_units[i], return_sequences=return_sequences))
            self.model.add(Dropout(dropout_rate))

        # Output layer
        self.model.add(Dense(units=1))

        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mean_squared_error',
            metrics=['mean_absolute_error']
        )

        print("✅ Model built successfully!")
        print(f"📋 Architecture: {len(lstm_units)} LSTM layers with {lstm_units} units")

        return self.model

    def train_model(self, epochs=50, batch_size=32, validation_split=0.1, verbose=1):
        """
        Train the LSTM model
        """
        print(f"🚀 Training model for {epochs} epochs...")

        history = self.model.fit(
            self.X_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=verbose,
            shuffle=False  # Important for time series
        )

        print("✅ Training completed!")
        return history

    def evaluate_model(self):
        """
        Evaluate model performance and make predictions
        """
        print("📊 Evaluating model performance...")

        # Make predictions
        y_pred_scaled = self.model.predict(self.X_test, verbose=0)

        # Inverse transform predictions to original scale
        y_test_orig = self.inverse_transform_predictions(self.y_test)
        y_pred_orig = self.inverse_transform_predictions(y_pred_scaled.flatten())

        # Calculate metrics
        mse = mean_squared_error(y_test_orig, y_pred_orig)
        mae = mean_absolute_error(y_test_orig, y_pred_orig)
        rmse = np.sqrt(mse)

        # Calculate directional accuracy
        actual_direction = np.diff(y_test_orig) > 0
        pred_direction = np.diff(y_pred_orig) > 0
        directional_accuracy = np.mean(actual_direction == pred_direction) * 100

        # Calculate percentage error
        mape = np.mean(np.abs((y_test_orig - y_pred_orig) / y_test_orig)) * 100

        print("\n📈 Model Performance:")
        print(f"   RMSE: ${rmse:.2f}")
        print(f"   MAE:  ${mae:.2f}")
        print(f"   MAPE: {mape:.2f}%")
        print(f"   Directional Accuracy: {directional_accuracy:.1f}%")

        return {
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'directional_accuracy': directional_accuracy,
            'predictions': y_pred_orig,
            'actual': y_test_orig
        }

    def inverse_transform_predictions(self, scaled_predictions):
        """
        Convert scaled predictions back to original price scale
        """
        # Create dummy array for inverse transform
        dummy = np.zeros((len(scaled_predictions), self.scaler.scale_.shape[0]))
        dummy[:, 0] = scaled_predictions

        # Inverse transform and return first column (Close price)
        return self.scaler.inverse_transform(dummy)[:, 0]

    def plot_results(self, results, last_n_days=100):
        """
        Create visualization of predictions vs actual prices
        """
        print("📊 Creating visualizations...")

        actual = results['actual']
        predictions = results['predictions']

        # Limit to last N days for clarity
        if len(actual) > last_n_days:
            actual = actual[-last_n_days:]
            predictions = predictions[-last_n_days:]

        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Stock Price Prediction Results', fontsize=16, fontweight='bold')

        # Plot 1: Actual vs Predicted prices
        axes[0, 0].plot(actual, label='Actual Price', color='blue', linewidth=2)
        axes[0, 0].plot(predictions, label='Predicted Price', color='red', linewidth=2, alpha=0.8)
        axes[0, 0].set_title('Price Prediction Comparison')
        axes[0, 0].set_xlabel('Days')
        axes[0, 0].set_ylabel('Price ($)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Prediction Error
        error = actual - predictions
        axes[0, 1].plot(error, color='green', linewidth=1)
        axes[0, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[0, 1].set_title('Prediction Error')
        axes[0, 1].set_xlabel('Days')
        axes[0, 1].set_ylabel('Error ($)')
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Error Distribution
        axes[1, 0].hist(error, bins=30, alpha=0.7, color='purple', edgecolor='black')
        axes[1, 0].axvline(x=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 0].set_title('Error Distribution')
        axes[1, 0].set_xlabel('Error ($)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Scatter plot
        axes[1, 1].scatter(actual, predictions, alpha=0.6, color='orange')

        # Add perfect prediction line
        min_val = min(actual.min(), predictions.min())
        max_val = max(actual.max(), predictions.max())
        axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)

        axes[1, 1].set_title('Actual vs Predicted Scatter')
        axes[1, 1].set_xlabel('Actual Price ($)')
        axes[1, 1].set_ylabel('Predicted Price ($)')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def predict_future(self, days=5):
        """
        Predict future stock prices
        """
        print(f"🔮 Predicting next {days} days...")

        # Get last sequence from test data
        last_sequence = self.X_test[-1].reshape(1, self.X_test.shape[1], self.X_test.shape[2])

        future_predictions = []

        for _ in range(days):
            # Predict next day
            next_pred = self.model.predict(last_sequence, verbose=0)[0, 0]
            future_predictions.append(next_pred)

            # Update sequence for next prediction
            # Shift sequence and add new prediction
            new_sequence = np.roll(last_sequence, -1, axis=1)
            new_sequence[0, -1, 0] = next_pred  # Update close price
            last_sequence = new_sequence

        # Convert to original scale
        future_prices = self.inverse_transform_predictions(np.array(future_predictions))

        print("\n🔮 Future Predictions:")
        for i, price in enumerate(future_prices, 1):
            print(f"   Day +{i}: ${price:.2f}")

        return future_prices


# ====================================================================
# MAIN EXECUTION FUNCTION
# ====================================================================

def run_ml_project(csv_file=None):
    """
    Main function to run the complete ML pipeline
    """
    print("=" * 60)
    print("🚀 AI/ML PROJECT: STOCK PRICE PREDICTION")
    print("=" * 60)

    # Initialize predictor
    predictor = StockPricePredictor()

    # Step 1: Load Data
    if csv_file:
        predictor.load_csv_data(csv_file)
    else:
        predictor.load_sample_data()

    # Step 2: Create Features
    predictor.create_features()

    # Step 3: Prepare Data
    features_to_use = ['Close', 'SMA_5', 'SMA_20', 'RSI', 'Volatility']
    predictor.prepare_data(
        sequence_length=60,
        features=features_to_use,
        test_split=0.2
    )

    # Step 4: Build Model
    predictor.build_model(lstm_units=[50, 25], dropout_rate=0.2)

    # Step 5: Train Model
    history = predictor.train_model(epochs=30, batch_size=32)

    # Step 6: Evaluate Model
    results = predictor.evaluate_model()

    # Step 7: Visualize Results
    predictor.plot_results(results)

    # Step 8: Future Predictions
    future_prices = predictor.predict_future(days=5)

    print("\n" + "=" * 60)
    print("✅ PROJECT COMPLETED SUCCESSFULLY!")
    print("=" * 60)

    return predictor, results


# ====================================================================
# USAGE EXAMPLES
# ====================================================================

if __name__ == "__main__":
    # Example 1: Run with sample data
    print("📊 Running with sample data...")
    predictor, results = run_ml_project()

    # Example 2: Run with your own CSV file
    # Uncomment the line below and provide your CSV file path
    # predictor, results = run_ml_project("your_stock_data.csv")

    # Display key results for interview discussion
    print(f"\n🎯 KEY RESULTS FOR INTERVIEW:")
    print(f"   • Model Type: LSTM Neural Network")
    print(f"   • Features Used: 5 technical indicators")
    print(f"   • Prediction Accuracy: {results['directional_accuracy']:.1f}%")
    print(f"   • Average Error: ${results['mae']:.2f}")
    print(f"   • Technologies: Python, TensorFlow, Scikit-learn")