# Stock Prediction - Bi-LSTM 7-Day Forecast


A Deep Learning project to predict next 7 days stock prices using Bidirectional LSTM.

### 📂 Files in this Repo
| File | Description |
|------|-------------|
| `bilstm_7day.h5` | Trained Bi-LSTM model (Input shape: 60, 3) |
| `dlphase3.ipynb` | Model training notebook |
| `app.py` | Streamlit web app |
| `requirements.txt` | Dependencies |

### 🧠 Model Details
- Architecture: Bidirectional LSTM
- Input: 60 days (Close, High, Low)
- Output: Next 7 days Close Price
- Model Loaded: `(None, 60, 3)` - Fixed `mse` error

### 🚀 How to Run
```bash
pip install -r requirements.txt
streamlit run final_app.py
