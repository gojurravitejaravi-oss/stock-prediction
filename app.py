import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Stock Prediction Bi-LSTM", layout="wide")
st.title("📈 Stock Price Prediction - Bi-LSTM (7-Day)")


@st.cache_resource
def load_bilstm():
    model = load_model("bilstm_7day.h5", compile=False)
    return model

model = load_bilstm()
st.success(f"Model Loaded: {model.input_shape}")

ticker = st.text_input("Enter Stock Ticker (e.g., RELIANCE.NS, TCS.NS, AAPL)", "RELIANCE.NS")

if st.button("Predict Next 7 Days"):
    with st.spinner("Downloading data..."):
        df = yf.download(ticker, period="2y")
        if df.empty:
            st.error("Invalid ticker or no data.")
            st.stop()

        data = df[['Close']].values
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(data)

        last_60 = scaled[-60:]
        last_60 = last_60.reshape(1, 60, 1)
        # If your model trained on 3 features, it expects 3. If 1 feature, it still works with this fix:
        if model.input_shape[2] == 3:
            last_60 = np.repeat(last_60, 3, axis=2)

        future_preds = []
        curr = last_60.copy()
        for i in range(7):
            pred = model.predict(curr, verbose=0)
            future_preds.append(pred[0,0])
            # slide window
            new_val = np.array([[[pred[0,0]]*curr.shape[2]]])
            curr = np.concatenate((curr[:,1:,:], new_val), axis=1)

        future_preds = scaler.inverse_transform(np.array(future_preds).reshape(-1,1))

        # FIXED DATES - No more 1970 bug
        last_date = df.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=7, freq='B')

        # Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index[-60:], y=df['Close'][-60:], name="Last 60 Days Close"))
        fig.add_trace(go.Scatter(x=future_dates, y=future_preds.flatten(), name="Predicted Next 7 Days", mode='markers+lines', line=dict(color='red')))
        st.plotly_chart(fig, use_container_width=True)

        pred_df = pd.DataFrame({"Date": future_dates, "Predicted Close": future_preds.flatten()})
        st.table(pred_df)
