import yfinance as yf
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from streamlit_option_menu import option_menu
import json
import plotly.graph_objects as go
from streamlit_lottie import st_lottie  # Import st_lottie
from tensorflow.keras.models import load_model  # Import load_model
from tensorflow.keras.utils import register_keras_serializable  # Import register_keras_serializable

st.set_page_config(layout="wide")

try:
    model = load_model('lstm_model.h5')
except Exception as e:
    st.error(f"Error loading LSTM model: {e}")

try:
    gru_model = load_model('gru_model.keras')
except Exception as e:
    st.error(f"Error loading GRU model: {e}")

@register_keras_serializable()
def mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

try:
    nbeats_model = load_model('nbeats_model.h5', custom_objects={'mse': mse})
except Exception as e:
    st.error(f"Error loading N-BEATS model: {e}")

# Load Lottie Animation from local file
with open("stock.json", "r", encoding="utf-8") as f:
    lottie_animation = json.load(f)

col1, col2 = st.columns([2, 2])

with col1:
    st.title("Stock Market App")
    st.subheader("Stock Market Analysis and Prediction Tool")
    st.write("""
    This app provides tools for analyzing and predicting stock market prices. 
    It leverages machine learning models like LSTM and GRU for making predictions and includes various technical indicators to assist in stock analysis. The app aims to help users make informed decisions by providing comprehensive insights into stock market trends.
    """)
    ticker = st.text_input("Enter Stock Ticker", "AAPL")

with col2:
    st_lottie(lottie_animation, height=320, key="stock_animation")

# Fetch the stock data
stock_data = yf.Ticker(ticker)
end_date = datetime.today().strftime('%Y-%m-%d')
stock_df = stock_data.history(period="1d", start="2024-1-1", end=end_date)

selected = option_menu(
    menu_title=None,  # No menu title
    options=["Stock Info", "Algorithms", "Indicators"],
    icons=["bar-chart", "code", "activity"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal"
)

# Display the stock data
if selected == "Stock Info":
    st.subheader(f"Stock data for {ticker}")
    st.write(stock_data.info.get("longBusinessSummary", "Long business summary is not available"))
    info_options = ["sector", "industry", "fullTimeEmployees", "website", "marketCap", "shortName", "longName", "exchange", "quoteType", "currency"]
    selected_info = st.selectbox("Select Information to Display", info_options)
    if stock_data.info and selected_info in stock_data.info:
        info_value = stock_data.info.get(selected_info, "Information is not available")
    else:
        info_value = "Information is not available"
    st.write(info_value)
    st.write("---")
    stock_df2 = stock_data.history(period="1d", interval="5m")
    st.subheader("Stock Price Candlestick Chart (5-minute intervals)")

    fig = go.Figure(data=[go.Candlestick(
        x=stock_df2.index,
        open=stock_df2['Open'],
        high=stock_df2['High'],
        low=stock_df2['Low'],
        close=stock_df2['Close'],
        name='5-minute intervals'
    )])

    fig.update_layout(
        width=1000,
        height=600,
        xaxis_title='Time',
        yaxis_title='Price',
        title=f'{ticker} Stock Price (5-minute intervals)',
        xaxis_rangeslider_visible=False
    )

    st.plotly_chart(fig)
    st.subheader("Stock Data")
    st.write("---")
    st.write(stock_df)

# Preprocess the data for the models
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

# Make predictions with N-BEATS model
X_nbeats = X.reshape(X.shape[0], time_step)
nbeats_predictions = nbeats_model.predict(X_nbeats)
nbeats_predictions = scaler.inverse_transform(nbeats_predictions)

# Display the predictions
if selected == "Algorithms":
    if lstm_predictions.size > 0 and gru_predictions.size > 0 and nbeats_predictions.size > 0:
        pred_df = pd.DataFrame({
            'LSTM Predicted Close': lstm_predictions.flatten(),
            'GRU Predicted Close': gru_predictions.flatten(),
            'N-BEATS Predicted Close': nbeats_predictions.flatten()
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
        fig.add_trace(go.Scatter(x=actual_df['Date'], y=pred_df['N-BEATS Predicted Close'], mode='lines', name='N-BEATS Predicted Close'))
        
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
    future_steps = st.number_input("Enter number of future steps to predict", min_value=1, max_value=20, value=2)
    last_data = scaled_data[-time_step:]
    future_predictions_lstm = []
    future_predictions_gru = []
    future_predictions_nbeats = []

    for _ in range(future_steps):
        lstm_pred = model.predict(last_data.reshape(1, time_step, 1))
        gru_pred = gru_model.predict(last_data.reshape(1, time_step, 1))
        nbeats_pred = nbeats_model.predict(last_data.reshape(1, time_step))

        future_predictions_lstm.append(lstm_pred[0, 0])
        future_predictions_gru.append(gru_pred[0, 0])
        future_predictions_nbeats.append(nbeats_pred[0, 0])

        last_data = np.append(last_data[1:], lstm_pred[0, 0].reshape(-1, 1), axis=0)

    future_predictions_lstm = scaler.inverse_transform(np.array(future_predictions_lstm).reshape(-1, 1))
    future_predictions_gru = scaler.inverse_transform(np.array(future_predictions_gru).reshape(-1, 1))
    future_predictions_nbeats = scaler.inverse_transform(np.array(future_predictions_nbeats).reshape(-1, 1))

    future_dates = pd.date_range(start=stock_df.index[-1], periods=future_steps + 1, inclusive='right')

    future_df = pd.DataFrame({
        'Date': future_dates,
        'LSTM Future Prediction': future_predictions_lstm.flatten(),
        'GRU Future Prediction': future_predictions_gru.flatten(),
        'N-BEATS Future Prediction': future_predictions_nbeats.flatten()
    })

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=future_df['Date'], y=future_df['LSTM Future Prediction'], mode='lines', name='LSTM Future Prediction'))
    fig.add_trace(go.Scatter(x=future_df['Date'], y=future_df['GRU Future Prediction'], mode='lines', name='GRU Future Prediction'))
    fig.add_trace(go.Scatter(x=future_df['Date'], y=future_df['N-BEATS Future Prediction'], mode='lines', name='N-BEATS Future Prediction'))

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

    def rsi(data, window=14):
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def stochastic_oscillator(data, window=14):
        low_min = data['Low'].rolling(window=window).min()
        high_max = data['High'].rolling(window=window).max()
        return 100 * (data['Close'] - low_min) / (high_max - low_min)

    def bollinger_bands(data, window=20):
        sma = data.rolling(window=window).mean()
        std = data.rolling(window=window).std()
        upper_band = sma + (std * 2)
        lower_band = sma - (std * 2)
        return upper_band, lower_band

    window_size = 20
    stock_df['SMA'] = moving_average(stock_df['Close'], window_size)
    stock_df['RSI'] = rsi(stock_df['Close'])
    stock_df['Stochastic'] = stochastic_oscillator(stock_df)
    stock_df['Upper Band'], stock_df['Lower Band'] = bollinger_bands(stock_df['Close'])

    stock_df['VWAP'] = (stock_df['Close'] * stock_df['Volume']).cumsum() / stock_df['Volume'].cumsum()
    stock_df['OBV'] = (np.sign(stock_df['Close'].diff()) * stock_df['Volume']).fillna(0).cumsum()
    exp1 = stock_df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = stock_df['Close'].ewm(span=26, adjust=False).mean()
    stock_df['MACD'] = exp1 - exp2
    stock_df['High-Low'] = stock_df['High'] - stock_df['Low']
    stock_df['ChaikinVolatility'] = stock_df['High-Low'].ewm(span=10).mean()

    indicators = ['Close', 'SMA', 'RSI', 'Stochastic', 'Upper Band', 'Lower Band', 'VWAP', 'OBV', 'MACD', 'ChaikinVolatility']
    selected_indicators = st.multiselect("Select up to 4 indicators", indicators, default=['SMA', 'RSI', 'Stochastic', 'Upper Band'])

    if len(selected_indicators) > 4:
        st.error("Please select up to 4 indicators only.")
    else:
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

        for indicator in selected_indicators:
            fig.add_trace(go.Scatter(x=actual_df['Date'], y=stock_df[indicator], mode='lines', name=indicator))

        fig.update_layout(
            width=1000,
            height=600,
            xaxis_title='Date',
            yaxis_title='Price',
            title='Stock Prices with Indicators'
        )

        st.plotly_chart(fig)
