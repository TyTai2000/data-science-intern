import pickle
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the model
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    age = int(request.form['age'])
    prediction = model.predict([[age]])
    value = prediction[0]

    return render_template('index.html', prediction=value, age=age)


if __name__ == '__main__':
    app.run(debug=True)
