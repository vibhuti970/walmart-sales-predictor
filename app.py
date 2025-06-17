from flask import Flask, request, render_template, redirect, flash, url_for
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')  # Prevent GUI issues on macOS
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import joblib

app = Flask(__name__)
app.secret_key = 'your_secret_key'

UPLOAD_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load trained model
model = joblib.load('rf_sales_model.pkl')

# Expected feature columns (must match training)
feature_columns = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Holiday_Flag']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        flash('No file part in request')
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        flash('No file selected')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        try:
            df = pd.read_csv(file)

            for col in feature_columns + ['Weekly_Sales']:
                if col not in df.columns:
                    return f"Missing required column: {col}"

            X_input = df[feature_columns]
            df['Predicted_Sales'] = model.predict(X_input)

            if not os.path.exists(UPLOAD_FOLDER):
                os.makedirs(UPLOAD_FOLDER)

            result_csv_path = os.path.join(UPLOAD_FOLDER, 'predictions.csv')
            df.to_csv(result_csv_path, index=False)

            plt.figure(figsize=(10, 5))
            plt.plot(df['Weekly_Sales'].values, label='Actual Sales', color='blue')
            plt.plot(df['Predicted_Sales'].values, label='Predicted Sales', color='orange')
            plt.legend()
            plt.title("Actual vs Predicted Sales")
            plt.xlabel("Time (rows)")
            plt.ylabel("Sales")
            plot_path = os.path.join(UPLOAD_FOLDER, 'sales_plot.png')
            plt.savefig(plot_path)
            plt.close()

            r2 = r2_score(df['Weekly_Sales'], df['Predicted_Sales'])
            mae = mean_absolute_error(df['Weekly_Sales'], df['Predicted_Sales'])
            rmse = np.sqrt(mean_squared_error(df['Weekly_Sales'], df['Predicted_Sales']))

            # ✅ Fix table output: render HTML table directly
            table_html = df.to_html(classes='data', header="true", index=False)

            return render_template('result.html',
                                   table=table_html,
                                   image=plot_path,
                                   download_link=result_csv_path,
                                   r2=round(r2, 4),
                                   mae=round(mae, 2),
                                   rmse=round(rmse, 2))

        except Exception as e:
            return f"Error processing file: {str(e)}"
    else:
        return "Invalid file format. Please upload a .csv file."

if __name__ == '__main__':
    app.run(debug=True)
