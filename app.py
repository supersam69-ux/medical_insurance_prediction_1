from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load the saved model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
    

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input from the form
    age = int(request.form['age'])
    sex = int(request.form['sex'])
    bmi = float(request.form['bmi'])
    children = int(request.form['children'])
    smoker = int(request.form['smoker'])
    region = int(request.form['region'])

    # Make prediction
    input_data = np.array([[age, sex, bmi, children, smoker, region]])
    prediction = model.predict(input_data)

    return render_template('index.html', prediction_text='The estimated insurance cost is USD {:.2f}'.format(prediction[0]))

if __name__ == "__main__":
    app.run(debug=True)
