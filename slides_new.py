import streamlit as st
import pandas as pd
from itables.streamlit import interactive_table

# Read csv
conv1d_train_TRH = pd.read_csv('slides_new/Conv1D Train temp & RH.csv')
conv1d_test_TRH = pd.read_csv('slides_new/Conv1D Test temp & RH.csv')

gru_train_TRH = pd.read_csv('slides_new/GRU Train temp & RH.csv')
gru_test_TRH = pd.read_csv('slides_new/GRU Test temp & RH.csv')

lstm_train_TRH = pd.read_csv('slides_new/LSTM Train Temp & RH.csv')
lstm_test_TRH = pd.read_csv('slides_new/LSTM Test Temp & RH.csv')

conv1d_model_loss = pd.read_csv('slides_new/conv1d_model_loss_df.csv')
gru_model_loss = pd.read_csv('slides_new/gru_model_loss_df.csv')
lstm_model_loss = pd.read_csv('slides_new/lstm_model_loss_df.csv')

metrics = pd.read_csv('slides_new/metrics.csv')
metrics_temp = pd.read_csv('slides_new/metrics_temp.csv')
metrics_rh = pd.read_csv('slides_new/metrics_rh.csv')

csv_train = [conv1d_train_TRH, gru_train_TRH, lstm_train_TRH]
csv_test = [conv1d_test_TRH, gru_test_TRH, lstm_test_TRH]
csv_loss = [conv1d_model_loss, gru_model_loss, lstm_model_loss]

def make_train_df(csv):
  csv = pd.DataFrame({
  "Temp Train Actuals": csv['Temperature Actuals'],
  "Temp Train Predictions": csv['Temperature Predictions'],
  "Rel Humidity Train Actuals": csv['Relative Humidity Actuals'],
  "Rel Humidity Train Predictions": csv['Relative Humidity Predictions']
})
  return csv
  
def make_test_df(csv):
  csv = pd.DataFrame({
  "Temp Test Actuals": csv['Temperature Actuals'],
  "Temp Test Predictions": csv['Temperature Predictions'],
  "Rel Humidity Test Actuals": csv['Relative Humidity Actuals'],
  "Rel Humidity Test Predictions": csv['Relative Humidity Predictions']
})
  return csv
  
def make_loss_df(csv):
  csv = pd.DataFrame({
  "Loss": csv['loss'],
  "Validation Loss": csv['val_loss'],
})
  return csv

def make_residual_plot_df(csv1, csv2, csv3):
  csv = pd.DataFrame({
    "Temperature Test Predictions": csv1['Temperature Predictions'],
    "LSTM Residuals": csv1['Temperature Actuals'] - csv1['Temperature Predictions'],
    "Conv1d Residuals": csv2['Temperature Actuals'] - csv2['Temperature Predictions'],
    "GRU Residuals": csv3['Temperature Actuals'] - csv3['Temperature Predictions'],
  })
  return csv

def make_residual_plot_df2(csv1, csv2, csv3):
  csv = pd.DataFrame({
    "Relative Humidity Test Predictions": csv1['Relative Humidity Predictions'],
    "LSTM Residuals": csv1['Relative Humidity Actuals'] - csv1['Relative Humidity Predictions'],
    "Conv1d Residuals": csv2['Relative Humidity Actuals'] - csv2['Relative Humidity Predictions'],
    "GRU Residuals": csv3['Relative Humidity Actuals'] - csv3['Relative Humidity Predictions'],
  })
  return csv

st.write("# Key Metrics and Performance")
st.write("## Model Metrics")

# Allot columns
colA, colB = st.columns(2)

with colA:
  st.write("### Temperature")
  interactive_table(metrics_temp)

with colB:
  st.write("### Relative Humidity")
  interactive_table(metrics_rh)

st.write("# Visualizing the Results")

st.write("## Train and Validation Loss Graphs")

# Allot columns
col1, col2, col3 = st.columns(3)

with col1:
  # LSTM Train and Validation Loss
  st.write("### LSTM Train and Validation Loss")
  st.line_chart(make_loss_df(lstm_model_loss), color=["#D760EE", "#01B5E7"])

with col2:
  # Conv1d Train and Validation Loss
  st.write("### Conv1d Train and Validation Loss")
  st.line_chart(make_loss_df(conv1d_model_loss), color=["#D760EE", "#01B5E7"])

with col3:
  # GRU Train and Validation Loss
  st.write("### GRU Train and Validation Loss")
  st.line_chart(make_loss_df(gru_model_loss), color=["#D760EE", "#01B5E7"])

st.write("## Actual vs Prediction Graphs")

# Allot columns
col4, col5 = st.columns(2)

with col4:
  # LSTM Train and Test Results
  st.write("### LSTM Train Results")
  st.line_chart(make_train_df(lstm_train_TRH), color=["#D760EE", "#01B5E7", "#D760EE", "#01B5E7"])

with col5:
  st.write("### LSTM Test Results")
  st.line_chart(make_test_df(lstm_test_TRH), color=["#D760EE", "#01B5E7", "#D760EE", "#01B5E7"])

# Allot columns
col6, col7 = st.columns(2)

with col6:
  # Conv1d Train and Test Results
  st.write("### Conv1d Train Results")
  st.line_chart(make_train_df(conv1d_train_TRH), color=["#D760EE", "#01B5E7", "#D760EE", "#01B5E7"])

with col7:
  st.write("### Conv1d Test Results")
  st.line_chart(make_test_df(conv1d_test_TRH), color=["#D760EE", "#01B5E7", "#D760EE", "#01B5E7"])

# Allot columns
col8, col9 = st.columns(2)

with col8:
  # GRU Train and Test Results
  st.write("### GRU Train Results")
  st.line_chart(make_train_df(gru_train_TRH), color=["#D760EE", "#01B5E7", "#D760EE", "#01B5E7"])

with col9:
  st.write("### GRU Test Results")
  st.line_chart(make_test_df(gru_test_TRH), color=["#D760EE", "#01B5E7", "#D760EE", "#01B5E7"])

st.write("## Residual Plots")

# Allot columns
col10, col11 = st.columns(2)

with col10:
  # Temperature Residual Plots
  st.write("### Temperature Residual Plot")
  st.scatter_chart(make_residual_plot_df(lstm_test_TRH, conv1d_test_TRH, gru_test_TRH), x='Temperature Test Predictions', y=['LSTM Residuals', 'Conv1d Residuals', 'GRU Residuals'], color=["#D760EE", "#01B5E7", "#7049E0"])

with col11:
  # Relative Humidity Residual Plots
  st.write("### Relative Humidity Residual Plot")
  st.scatter_chart(make_residual_plot_df2(lstm_test_TRH, conv1d_test_TRH, gru_test_TRH), x='Relative Humidity Test Predictions', y=['LSTM Residuals', 'Conv1d Residuals', 'GRU Residuals'], color=["#D760EE", "#01B5E7", "#7049E0"])

