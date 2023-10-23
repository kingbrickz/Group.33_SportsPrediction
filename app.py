from flask import Flask, render_template, request, redirect
import joblib
import numpy as np

app = Flask(__name__)

# Load your trained machine learning model
# Replace 'your_model.pkl' with the actual filename of your model
model = joblib.load('xgb_22.pkl')

# Define the route for the home page
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get user input from the form
        try:
            movement_reactions = float(request.form['movement_reactions'])
            passing = float(request.form['passing'])
            mentality_composure = float(request.form['mentality_composure'])
            dribbling = float(request.form['dribbling'])
            potential = float(request.form['potential'])
            release_clause_eur = float(request.form['release_clause_eur'])
            wage_eur = float(request.form['wage_eur'])
            value_eur = float(request.form['value_eur'])
            power_shot_power = float(request.form['power_shot_power'])
        except ValueError:
            return "Invalid input. Please enter numeric values."

        # Perform prediction using your model
        input_data = np.array([movement_reactions, passing, mentality_composure, dribbling, potential,
                               release_clause_eur, wage_eur, value_eur, power_shot_power]).reshape(1, -1)
        prediction = model.predict(input_data)[0]

        # Render the result template with the prediction
        return render_template('result.html', prediction=prediction)

    return render_template('index.html')

# Define the route for the /predict URL
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get user input from the form
        try:
            movement_reactions = float(request.form['movement_reactions'])
            passing = float(request.form['passing'])
            mentality_composure = float(request.form['mentality_composure'])
            dribbling = float(request.form['dribbling'])
            potential = float(request.form['potential'])
            release_clause_eur = float(request.form['release_clause_eur'])
            wage_eur = float(request.form['wage_eur'])
            value_eur = float(request.form['value_eur'])
            power_shot_power = float(request.form['power_shot_power'])
        except ValueError:
            return "Invalid input. Please enter numeric values."

        # Perform prediction using your model
        input_data = np.array([movement_reactions, passing, mentality_composure, dribbling, potential,
                               release_clause_eur, wage_eur, value_eur, power_shot_power]).reshape(1, -1)
        prediction = model.predict(input_data)[0]

        # Render the result template with the prediction
        return render_template('result.html', prediction=prediction)

    # If it's a GET request, you can handle it as needed, or you can simply redirect to the home page
    return redirect('/')

if __name__ == '__main__':
    app.run(debug=True)
