import yfinance as yf
import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import time
from datetime import datetime
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
import requests
import json


model = load_model('lstm_model.h5')
gru_model = load_model('gru_model.keras')

st.set_page_config(layout="wide")
# Load Lottie Animation from local file
with open("stock.json", "r", encoding="utf-8") as f:
    lottie_animation = json.load(f)

st.title("Real-Time Stock Prediction App")

col1, col2 = st.columns([2, 2])
with col1:
    
    st.subheader("Stock Market Analysis and Prediction Tool")
    st.write("""
    This app provides tools for analyzing and predicting stock market prices. 
    It leverages machine learning models like LSTM and GRU for making predictions and includes various technical indicators to assist in stock analysis. The app aims to help users make informed decisions by providing comprehensive insights into stock market trends.
    """)
    ticker = st.text_input("Enter Stock Ticker", "AAPL")

with col2:
    st_lottie(lottie_animation, height=400, key="stock_animation")

# Fetch the stock data
stock_data = yf.Ticker(ticker)
end_date = datetime.today().strftime('%Y-%m-%d')
stock_df = stock_data.history(period="1d", start="2024-1-1", end=end_date)

selected = option_menu(
    menu_title=None,  # No menu title
    options=["Stock Info", "LSTM & GRU", "Indicators"],
    icons=["bar-chart", "line-chart", "activity"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal"
)

# Display the stock data
if selected == "Stock Info":
    st.subheader(f"Stock data for {ticker}")
    st.write(stock_data.info["longBusinessSummary"])
    st.write(stock_data.info["sector"])
    st.write("---")
    st.subheader("Stock Price Line Chart")
    st.line_chart(stock_df['Close'])
    st.subheader("Stock Data")
    st.write("---")
    st.write(stock_df)

# Preprocess the data for the LSTM model
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(stock_df['Close'].values.reshape(-1, 1))

# Prepare the data for prediction
def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        a = data[i:(i + time_step), 0]
        X.append(a)
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

time_step = 10
X, Y = create_dataset(scaled_data, time_step)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Make predictions with LSTM model
lstm_predictions = model.predict(X)
lstm_predictions = scaler.inverse_transform(lstm_predictions)

# Make predictions with GRU model
gru_predictions = gru_model.predict(X)
gru_predictions = scaler.inverse_transform(gru_predictions)

# Display the predictions
import plotly.graph_objects as go

if selected == "LSTM & GRU":
    if lstm_predictions.size > 0 and gru_predictions.size > 0:
        pred_df = pd.DataFrame({
            'LSTM Predicted Close': lstm_predictions.flatten(),
            'GRU Predicted Close': gru_predictions.flatten()
        })
        actual_df = stock_df.reset_index()
        actual_df['Date'] = pd.to_datetime(actual_df['Date'])

        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=actual_df['Date'],
            open=stock_df['Open'],
            high=stock_df['High'],
            low=stock_df['Low'],
            close=stock_df['Close'],
            name='Actual Prices'
        ))
        fig.add_trace(go.Scatter(x=actual_df['Date'], y=pred_df['LSTM Predicted Close'], mode='lines', name='LSTM Predicted Close'))
        fig.add_trace(go.Scatter(x=actual_df['Date'], y=pred_df['GRU Predicted Close'], mode='lines', name='GRU Predicted Close'))
        
        fig.update_layout(
            width=1000,  
            height=700, 
            xaxis_title='Date',
            yaxis_title='Price',
            title='Actual and Predicted Stock Prices'
        )
        
        st.plotly_chart(fig)
    else:
        st.write("No predictions to display.")
        
    # Predict future prices for the next 30 days
    future_steps = 30
    last_data = scaled_data[-time_step:]
    future_predictions_lstm = []
    future_predictions_gru = []

    for _ in range(future_steps):
        lstm_pred = model.predict(last_data.reshape(1, time_step, 1))
        gru_pred = gru_model.predict(last_data.reshape(1, time_step, 1))

        future_predictions_lstm.append(lstm_pred[0, 0])
        future_predictions_gru.append(gru_pred[0, 0])

        last_data = np.append(last_data[1:], lstm_pred[0, 0].reshape(-1, 1), axis=0)

    future_predictions_lstm = scaler.inverse_transform(np.array(future_predictions_lstm).reshape(-1, 1))
    future_predictions_gru = scaler.inverse_transform(np.array(future_predictions_gru).reshape(-1, 1))

    future_dates = pd.date_range(start=stock_df.index[-1], periods=future_steps + 1, inclusive='right')

    future_df = pd.DataFrame({
        'Date': future_dates,
        'LSTM Future Prediction': future_predictions_lstm.flatten(),
        'GRU Future Prediction': future_predictions_gru.flatten()
    })

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=future_df['Date'], y=future_df['LSTM Future Prediction'], mode='lines', name='LSTM Future Prediction'))
    fig.add_trace(go.Scatter(x=future_df['Date'], y=future_df['GRU Future Prediction'], mode='lines', name='GRU Future Prediction'))

    fig.update_layout(
        width=1000,
        height=700,
        xaxis_title='Date',
        yaxis_title='Price',
        title='Future Stock Price Predictions'
    )

    st.plotly_chart(fig)

if selected == "Indicators":
    def moving_average(data, window_size):
        return data.rolling(window=window_size).mean()

    window_size = 20
    stock_df['SMA'] = moving_average(stock_df['Close'], window_size)

    stock_df['VWAP'] = (stock_df['Close'] * stock_df['Volume']).cumsum() / stock_df['Volume'].cumsum()

    stock_df['OBV'] = (np.sign(stock_df['Close'].diff()) * stock_df['Volume']).fillna(0).cumsum()

    exp1 = stock_df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = stock_df['Close'].ewm(span=26, adjust=False).mean()
    stock_df['MACD'] = exp1 - exp2

    stock_df['High-Low'] = stock_df['High'] - stock_df['Low']
    stock_df['ChaikinVolatility'] = stock_df['High-Low'].ewm(span=10).mean()

    indicators = ['Close', 'SMA', 'VWAP', 'OBV', 'MACD', 'ChaikinVolatility']
    stock_df['Average'] = stock_df[indicators].mean(axis=1)

    actual_df = stock_df.reset_index()

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=actual_df['Date'],
        open=stock_df['Open'],
        high=stock_df['High'],
        low=stock_df['Low'],
        close=stock_df['Close'],
        name='Actual Prices'
    ))

    fig.add_traces([go.Scatter(x=actual_df['Date'], y=stock_df['SMA'], mode='lines', name='SMA'),
                    go.Scatter(x=actual_df['Date'], y=stock_df['VWAP'], mode='lines', name='VWAP'),
                    go.Scatter(x=actual_df['Date'], y=stock_df['OBV'], mode='lines', name='OBV'),
                    go.Scatter(x=actual_df['Date'], y=stock_df['MACD'], mode='lines', name='MACD'),
                    go.Scatter(x=actual_df['Date'], y=stock_df['ChaikinVolatility'], mode='lines', name='Chaikin Volatility'),
                    go.Scatter(x=actual_df['Date'], y=stock_df['Average'], mode='lines', name='Average')])

    fig.update_layout(
        width=1000,
        height=600,
        xaxis_title='Date',
        yaxis_title='Price',
        title='Stock Prices with Indicators'
    )

    st.plotly_chart(fig)
