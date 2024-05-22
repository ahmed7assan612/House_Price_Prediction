import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import math

app = Flask(__name__)
model = pickle.load(open("KNN_model.pkl",'rb'))

@app.route('/')
def welcome():
    return 'Welcome All'

@app.route('/predict', methods=['Get'])
def predict():
    int_features  = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = round(prediction[0])
    return "Hello The answer is"+(prediction)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)