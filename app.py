import os
import io
import sys

import numpy as np
import pandas as pd
from flask import Flask, request, render_template, send_file, flash, redirect, url_for

from src.exception import CustomException
from src.pipeline.predict_pipeline import PredictPipeline

app = Flask(__name__)
app.secret_key = "threatforecaster_secret"

ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        flash('No file uploaded.')
        return redirect(url_for('index'))

    file = request.files['file']

    if file.filename == '':
        flash('No file selected.')
        return redirect(url_for('index'))

    if not allowed_file(file.filename):
        flash('Only CSV files are supported.')
        return redirect(url_for('index'))

    try:
        df = pd.read_csv(file)

        ids = df['id'] if 'id' in df.columns else pd.Series(range(len(df)), name='id')

        pipeline = PredictPipeline()
        predictions = pipeline.predict(df)

        results_df = pd.DataFrame({
            'id':     ids.values,
            'target': predictions
        })

        # Render results page
        table_html = results_df.head(50).to_html(
            classes='results-table', index=False, border=0
        )
        total      = len(results_df)
        infected   = int(predictions.sum())
        clean      = total - infected

        # Build downloadable CSV in memory
        csv_buffer = io.StringIO()
        results_df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        return render_template(
            'results.html',
            table=table_html,
            total=total,
            infected=infected,
            clean=clean,
            csv_data=csv_buffer.getvalue()
        )

    except Exception as e:
        flash(f'Error during prediction: {str(e)}')
        return redirect(url_for('index'))


@app.route('/download', methods=['POST'])
def download():
    csv_data = request.form.get('csv_data', '')
    buffer = io.BytesIO(csv_data.encode('utf-8'))
    buffer.seek(0)
    return send_file(
        buffer,
        mimetype='text/csv',
        as_attachment=True,
        download_name='submission.csv'
    )


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)