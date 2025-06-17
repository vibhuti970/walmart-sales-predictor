# walmart-sales-predictor
# Walmart Sales Forecasting Web App

This is a Flask web app that allows users to upload Walmart-style sales data in CSV format and get predictions using a trained Random Forest model.

## Features:
- Upload CSV with features like Temperature, Fuel_Price, etc.
- Predict `Weekly_Sales`
- Visualize actual vs predicted
- Download prediction results
- View model performance (R², MAE, RMSE)

## How to Run

```bash
pip install -r requirements.txt
python app.py
