from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the model
lin_reg_model = pickle.load(open('lin_reg_model.pkl', 'rb'))
svm_model = pickle.load(open('svm_model.pkl', 'rb'))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def predict():
    Temperature = float(request.form['Temperature'])
    Humidity = float(request.form['Humidity'])
    model_type = request.form['model_type']

    if model_type == 'linear_regression':
        # Use the Linear Regression model to make a prediction
        prediction = lin_reg_model.predict([[Temperature, Humidity]])
    else:
        # Use the Support Vector Machine model to make a prediction
        prediction = svm_model.predict([[Temperature, Humidity]])

    # Format the prediction to 4 digits after the comma
    prediction = f"{prediction[0]:.4f}"

    return render_template('index.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)