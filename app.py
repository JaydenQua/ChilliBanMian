import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error
import math
import os

st.set_page_config(layout="wide", page_title="LSTM Stock Price Predictor")

# Choose the stock
TICKER = 'MSFT'
START_DATE = '2010-01-01'
END_DATE = '2025-01-01'
TIME_STEP = 60 # Look-back window (60 days)
MODEL_PATH = 'lstm_stock_model.keras'


@st.cache_data
def load_and_preprocess_data(ticker, start, end, time_step):
    st.subheader(f"1. Downloading Stock Data for {ticker}...")
    
    # Get the data
    stock_data = yf.download(ticker, start=start, end=end)
    if stock_data.empty:
        st.error("Failed to download data.")
        return None, None, None, None, None, None, None

    stock_data = stock_data.dropna()
    data_close = stock_data['Close'].values.reshape(-1, 1)

    #Normalization?
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data_close)

    # Splitting data
    training_data_len = int(len(scaled_data) * 0.8)
    train_data = scaled_data[0:training_data_len, :]
    test_data = scaled_data[training_data_len - time_step:, :]
    
    # Store the actual close prices for evaluation later
    actual_prices = data_close[training_data_len:]
    test_dates = stock_data.index[training_data_len:].values

    def create_dataset(dataset, time_step):
        X, Y = [], []
        for i in range(time_step, len(dataset)):
            X.append(dataset[i-time_step:i, 0])
            Y.append(dataset[i, 0])
        return np.array(X), np.array(Y)

    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)

    # Reshape for LSTM
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    st.success("Data Preprocessing Complete!")
    return X_train, y_train, X_test, y_test, scaler, actual_prices, test_dates


@st.cache_resource
def train_and_save_model(X_train, y_train, X_test, model_path, epochs=10):
    
    # Check if model exists, load it to save time
    if os.path.exists(model_path):
        st.info(f"Loading pre-trained model from {model_path}...")
        model = load_model(model_path)
        return model

    st.subheader(f"2. Building and Training Model (Epochs={epochs})...")

    # Model Building
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    #Train
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Use st.spinner to show progress during the slow training
    with st.spinner("Training in progress..."):

        model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=32,
            verbose=0
        )

    print("--- Training Complete lol ---")
    
    # Save the trained model
    model.save(model_path)
    st.success(f"Training Complete and model saved to {model_path}!")
    
    return model


def main():
    st.title(f"Stock Price Prediction using LSTM ({TICKER})")
    st.info("This application uses a Long Short-Term Memory (LSTM) network to forecast stock prices. The model is trained on data from 2010 to present.")
    st.markdown("---")

    # 1. Data Acquisition and Preprocessing
    X_train, y_train, X_test, y_test, scaler, actual_prices, test_dates = load_and_preprocess_data(
        TICKER, START_DATE, END_DATE, TIME_STEP
    )
    
    if X_train is None:
        return

    # 2. Model Training
    model = train_and_save_model(X_train, y_train, X_test, MODEL_PATH, epochs=10)

    st.markdown("---")
    st.subheader("3. Model Prediction and Evaluation")
    
    #predict
    with st.spinner("Generating predictions..."):
        predictions = model.predict(X_test, verbose=0)
        # Convert back to USD price from normalization 0-1
        predictions = scaler.inverse_transform(predictions) 

    # Evaluation
    rmse = math.sqrt(mean_squared_error(actual_prices, predictions))
    
    # Second Evaluation??????
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Root Mean Squared Error (RMSE)", value=f"${rmse:,.2f}")
    with col2:
        st.metric(label="Predicted Ticker", value=TICKER)


    # Visualization
    st.subheader("4. Price Forecast Visualization")
    
    
    plt.close('all') 
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(test_dates, actual_prices, color='red', linewidth=2, label='Actual Price')
    ax.plot(test_dates, predictions, color='blue', linewidth=2, label='Predicted Price')

    ax.set_title(f'Stock Price Forecast vs. Actual ({TICKER})')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Close Price USD ($)', fontsize=12)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True)
    
    # Display the plot in Streamlit
    st.pyplot(fig)


if __name__ == '__main__':

    main()
