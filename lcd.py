import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

lcd = Flask(__name__)
model = pickle.load(open('modellcd.pkl', 'rb'))

@lcd.route('/')
def home():
  return render_template('lcd.html')

@lcd.route('/predict',methods=['POST'])
def predict():
  '''input_features = [int(x) for x in request.form.values()]
  features_value = [np.array(input_features)]

  features_name = ['GENDER', 'AGE', 'SMOKING',
       'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE',
       'CHRONIC DISEASE', 'FATIGUE', 'ALLERGY', 'WHEEZING', 'ALCOHOL CONSUMING', 'COUGHING', 
       'SHORTNESS OF BREATH', 'SWALLOWING DIFFICULTY', 'CHEST PAIN']

  df = pd.DataFrame(features_value, columns=features_name)
  output = modellcd.predict(df)'''
  gender=request.form.get('GENDER')
  age=request.form.get('AGE')
  smoking=request.form.get('SMOKING')
  yellow_fingers=request.form.get('YELLOW_FINGERS')
  anxiety=request.form.get('ANXIETY')
  peer_pressure=request.form.get('PEER_PRESSURE')
  chronic_disease=request.form.get('CHRONIC_DISEASE')
  fatigue=request.form.get('FATIGUE')
  allergy=request.form.get('ALLERGY')
  wheezing=request.form.get('WHEEZING')
  alcohol_consuming=request.form.get('ALCOHOL_CONSUMING')
  coughing=request.form.get('COUGHING')
  shortness_of_breath=request.form.get('SHORTNESS_OF_BREATH')
  swallowing_difficulty=request.form.get('SWALLOWING_DIFFICULTY')
  chest_pain=request.form.get('CHEST_PAIN')
  
  output=model.predict([[gender,age,smoking,yellow_fingers,anxiety,peer_pressure,chronic_disease,fatigue,allergy,wheezing,alcohol_consuming,coughing,shortness_of_breath,swallowing_difficulty,chest_pain]])

  if output == 1:
      res_val = "Lung cancer"
  else:
      res_val = "Not Lung cancer"


  return render_template('lcd.html', prediction_text='Patient might have {}'.format(res_val))

if __name__ == "__main__":
    lcd.run(debug=True)