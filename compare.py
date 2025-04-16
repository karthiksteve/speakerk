import matplotlib.pyplot as plt

# Define model names and metrics for comparison
models = ['ARIMA', 'Vanilla LSTM', 'GRU', 'LSTM + Autoencoder']
rmse = [712.34, 588.76, 553.22, 504.62]
mae = [602.19, 470.45, 436.92, 388.74]
mape = [4.87, 3.35, 2.98, 2.14]

# Set up bar chart
fig, ax = plt.subplots(1, 3, figsize=(15, 5))

# RMSE Bar Plot
ax[0].bar(models, rmse, color='skyblue')
ax[0].set_title("RMSE Comparison")
ax[0].set_ylabel("RMSE (MW)")
ax[0].tick_params(axis='x', rotation=15)

# MAE Bar Plot
ax[1].bar(models, mae, color='lightgreen')
ax[1].set_title("MAE Comparison")
ax[1].set_ylabel("MAE (MW)")
ax[1].tick_params(axis='x', rotation=15)

# MAPE Bar Plot
ax[2].bar(models, mape, color='salmon')
ax[2].set_title("MAPE Comparison")
ax[2].set_ylabel("MAPE (%)")
ax[2].tick_params(axis='x', rotation=15)

plt.suptitle("Model Performance Metrics Comparison", fontsize=14)
plt.tight_layout()
plt.subplots_adjust(top=0.88)

# Save chart
chart_path = "C:\\Users\\speak\\OneDrive\\Documents\\math_optimmization\\model_comparison_bar_chart.png"
plt.savefig(chart_path, dpi=300)
plt.show()

chart_path
