from flask import Flask, request, jsonify, render_template_string
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
from pathlib import Path

app = Flask(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / 'data' / 'processed' / 'dataset.csv'
MODEL_PATH = PROJECT_ROOT / 'model' / 'breast_cancer_model.pkl'


@app.route('/', methods=['GET'])
def home():
    return render_template_string("""
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Breast Cancer Prediction</title>
    <style>
      body { font: 16px system-ui, sans-serif; max-width: 640px; margin: 3rem auto; padding: 0 1rem; color: #202124; }
      h1 { color: #8b1e3f; }
      form { display: grid; gap: 1rem; }
      label { display: grid; gap: .35rem; }
      input, button { font: inherit; padding: .65rem; }
      button { color: white; background: #8b1e3f; border: 0; border-radius: 4px; cursor: pointer; }
      #result { margin-top: 1rem; font-weight: 600; }
    </style>
  </head>
  <body>
    <h1>Breast Cancer Prediction</h1>
    <p>Enter the feature values to get a model prediction.</p>
    <form id="prediction-form">
      <label>Tumor size
        <input name="tumor_size" type="number" step="any" value="2.5" required>
      </label>
      <label>Age
        <input name="age" type="number" step="any" value="45" required>
      </label>
      <label>Other features
        <input name="other_features" type="number" step="any" value="1.2" required>
      </label>
      <button type="submit">Get prediction</button>
    </form>
    <p id="result" role="status" aria-live="polite"></p>
    <script>
      document.getElementById('prediction-form').addEventListener('submit', async (event) => {
        event.preventDefault();
        const values = Object.fromEntries(new FormData(event.currentTarget));
        const result = document.getElementById('result');
        result.textContent = 'Calculating...';
        try {
          const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(Object.fromEntries(
              Object.entries(values).map(([key, value]) => [key, Number(value)])
            ))
          });
          const data = await response.json();
          if (!response.ok) throw new Error(data.error || 'Prediction request failed.');
          result.textContent = `Predicted likelihood: ${Number(data.prediction).toFixed(3)}`;
        } catch (error) {
          result.textContent = `Error: ${error.message}`;
        }
      });
    </script>
  </body>
</html>
    """)


# Load the dataset
def load_data():
    data = pd.read_csv(DATA_PATH)
    return data

# Train the model
def train_model():
    data = load_data()
    target_column = 'target' if 'target' in data.columns else 'cancer_likelihood'
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Save the model
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    return model

# Predict function
@app.route('/predict', methods=['POST'])
def predict():
    model = joblib.load(MODEL_PATH)
    input_data = request.json
    input_df = pd.DataFrame(input_data, index=[0])
    
    prediction = model.predict(input_df)
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    train_model()
    app.run(debug=True)