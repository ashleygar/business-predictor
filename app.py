import numpy as np
import pickle

from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LogisticRegression


app = Flask(__name__, template_folder='templates')
pipe = pickle.load(open('model/model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    args = request.form
    data = pd.DataFrame({
            'category': [args.get('category')],
            'license': [args.get('license')],
            'neighborhoods': [args.get('neighborhoods')],
            'years_in_business': [float(args.get('years_in_business'))]
        })
    success = pipe.predict(data)[0]
    probability = pipe.predict_proba(data)[0][1]
    if probability >= 0.17:
        details = 'Good news! This business has an above average chance of survival in 3 years!'
    if probability >=0.08:
        if probability  <0.17:
            details = 'Caution! This business has a less than average chance of survival in 3 years.'
    else:
        details = 'Beware! This business has an extremely low chance of survival in 3 years.'
    probability = pipe.predict_proba(data)[0][1]
    predict = np.prod([probability,100])
    prediction = int(predict)
    return render_template(
        'result.html',
        details=details,
        prediction=prediction)

if __name__ == '__main__':
    app.run(port=5500, debug=True)
