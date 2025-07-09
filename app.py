from flask import Flask, request, render_template 
import numpy as np
import joblib

app = Flask(__name__)

# Load your trained model and the scaler
model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")  # ✅ Load the scaler

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        avg_grade = float(request.form['avg_grade'])
        failures = int(request.form['failures'])
        studytime = int(request.form['studytime'])
        absences = int(request.form['absences'])
        goout = int(request.form['goout'])
        freetime = int(request.form['freetime'])
        internet = 1 if request.form['internet'] == 'Yes' else 0

        # Create feature array
        features = np.array([[avg_grade, failures, studytime, absences, goout, freetime, internet]])

        # ✅ Apply scaling
        scaled_features = scaler.transform(features)

        # Predict using the trained model
        prediction = model.predict(scaled_features)

        return render_template('index.html', prediction_text=f"Predicted Final Grade (G3): {round(prediction[0], 2)} / 20")

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {e}")

if __name__ == "__main__":
    app.run(debug=True)
