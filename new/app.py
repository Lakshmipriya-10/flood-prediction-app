from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)


with open('xgb_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    
    features = [
      
        float(request.form['TopographyDrainage']),
        float(request.form['RiverManagement']),
        float(request.form['ClimateChange']),
        float(request.form['Siltation']),
        float(request.form['AgriculturalPractices']),
        float(request.form['Encroachments']),
        float(request.form['IneffectiveDisasterPreparedness']),
        float(request.form['Landslides']),
        float(request.form['WetlandLoss']),
         float(request.form['PoliticalFactors']), 
    ]

    
    prediction = model.predict([features])[0]
    
    
    if prediction >= 0.7:
        risk_level = 'High Risk'
    elif prediction >= 0.4:
        risk_level = 'Medium Risk'
    else:
        risk_level = 'Low Risk'

    
    return render_template('result.html', risk=risk_level)

if __name__ == '__main__':
    app.run(debug=True)
