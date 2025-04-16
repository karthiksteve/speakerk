# ⚡ Dynamic Power Consumption Forecasting using LSTM and Autoencoder

This repository presents a hybrid deep learning approach for predicting short-term electricity consumption using a combination of Autoencoders and Long Short-Term Memory (LSTM) networks. The model is evaluated using the AEP Hourly Energy Consumption dataset and compared against traditional models like ARIMA and GRU.

---

## 📁 Project Structure

```
📦your-repo/
│
├── 📂data/
│   └── AEP_hourly.csv               # Dataset
│
├── 📂notebooks/
│   └── energy_forecasting.ipynb     # Jupyter Notebook with EDA and model training
│
├── 📊 forecast_plot.png             # Actual vs predicted plot
│
├── requirements.txt                 # List of required Python packages
└── README.md                        # This file
```

---

## 📊 Dataset

- **Name**: AEP Hourly Energy Consumption
- **Source**: UCI Machine Learning Repository
- **Duration**: 2004 to 2018 (hourly readings)
- **Target Variable**: `Value` (Power consumption in MW)

---

## 🧠 Methodology

1. **Data Preprocessing**: Conversion of timestamp, handling missing values, feature extraction (hour, day, month, etc.).
2. **Feature Engineering**: Creation of time-based features; optional lag features and rolling statistics.
3. **Autoencoder**: Compresses and denoises feature input before passing to LSTM.
4. **LSTM Network**: Learns temporal dependencies to forecast future consumption.
5. **Model Evaluation**: Uses RMSE, MAE, and MAPE for model comparison.

---

## 📈 Model Performance

| Model             | RMSE    | MAE     | MAPE   |
|------------------|---------|---------|--------|
| ARIMA            | 712.34  | 602.19  | 4.87%  |
| Vanilla LSTM     | 588.76  | 470.45  | 3.35%  |
| GRU              | 553.22  | 436.92  | 2.98%  |
| **Proposed Model** | **504.62** | **388.74** | **2.14%** |

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/power-forecast-lstm-ae.git
cd power-forecast-lstm-ae
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the notebook
Open and execute `notebooks/energy_forecasting.ipynb` in Jupyter or Google Colab.

---

## 📦 Requirements

- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- tensorflow / keras

---

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for more information.

---

## 🌟 Acknowledgments

- UCI Machine Learning Repository
- TensorFlow & Keras community
- IEEE research community and formatting support
