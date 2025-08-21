from flask import Flask, render_template, request
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model and label encoders
with open('titanic_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('le_sex.pkl', 'rb') as sex_file:
    sex_encoder = pickle.load(sex_file)

with open('le_embarked.pkl', 'rb') as embarked_file:
    embarked_encoder = pickle.load(embarked_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form

        # Extract and transform input data
        Pclass = int(data['Pclass'])
        Sex = sex_encoder.transform([data['Sex']])[0]
        Age = float(data['Age'])
        SibSp = int(data['SibSp'])
        Parch = int(data['Parch'])
        Fare = float(data['Fare'])
        Embarked = embarked_encoder.transform([data['Embarked']])[0]

        # Prepare feature array for prediction
        features = np.array([[Pclass, Sex, Age, SibSp, Parch, Fare, Embarked]])

        # Predict and return result
        prediction = model.predict(features)[0]
        result = 'Survived' if prediction == 1 else 'Did Not Survive'
        return render_template('index.html', result=result)

    except Exception as e:
        return render_template('index.html', result=f"Error: {e}")

if __name__ == '__main__':
    app.run(debug=True)
