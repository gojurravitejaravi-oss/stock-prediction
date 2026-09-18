import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

st.title("Day 57: Bi-LSTM 7-Day Stock Prediction")

@st.cache_resource
def get_model():
    return load_model(r"C:\Users\raviteja\bilstm_7day.h5", compile=False)

model = get_model()
st.success(f"Model Loaded: {model.input_shape}")

ticker = st.text_input("Ticker", "RELIANCE.NS")

if st.button("Predict"):
    df = yf.download(ticker, period="2y", auto_adjust=True)
    n_feat = model.input_shape[-1]
    if n_feat is None:
        n_feat = 3
    cols = ["Close", "High", "Low"]
    if n_feat == 1:
        data = df[["Close"]].values
    else:
        data = df[cols[:n_feat]].values
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)
    last = scaled[-60:].reshape(1, 60, n_feat)
    preds = []
    cur = last.copy()
    for i in range(7):
        p = model.predict(cur, verbose=0)
        val = float(p[0,0])
        preds.append(val)
        if n_feat == 1:
            next_row = np.array([[[val]]])
        else:
            next_feat = cur[0,-1,:].copy()
            next_feat[0] = val
            next_row = next_feat.reshape(1,1,n_feat)
        cur = np.append(cur[:,1:,:], next_row, axis=1)
    dummy = np.zeros((7, n_feat))
    dummy[:,0] = preds
    future = scaler.inverse_transform(dummy)[:,0]
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=df["Close"].tail(200), name="History"))
    fig.add_trace(go.Scatter(y=future, name="Next 7 Days", mode="lines+markers"))
    st.plotly_chart(fig, use_container_width=True)
    st.table(pd.DataFrame({"Day": range(1,8), "Predicted Close": future}))
    st.balloons()