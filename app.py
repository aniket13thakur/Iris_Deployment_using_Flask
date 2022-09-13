from flask import Flask,jsonify,request
import numpy as np
import joblib

import flask 
app = Flask(__name__)

#load the data
KNN_model = joblib.load("KNN_Model.pkl")

@app.route('/')
def home():
	return "Iris Data Prediction System Using ML"

@app.route('/index')
def index():
	return flask.render_template('index.html')


@app.route('/predict',methods = ['POST'])
def predict():
	

	try:
		
		#Get Data from UI
		to_predict_dict = request.form.to_dict()
		print('to_predict_dict :-',to_predict_dict)

		# Get Values from the dict
		to_predict_list = to_predict_dict.values()
		print('to_predict_list :- ',to_predict_list)

		# Convert list values into float
		to_predict_list_float = list(map(float,to_predict_list))
		print('to_predict_list_float :- ',to_predict_list_float)


		# Convert list  into np.array
		arr = np.array(to_predict_list_float)
		print('arr 1-D :- ',arr)

		# Convert 1_D array to 2_D
		arr = arr.reshape(1,-1)
		print('arr 2-D:- ',arr)

		print(KNN_model)

		#Predict the result
		prediction = KNN_model.predict(arr)
		print('Prediction :- ',prediction)


		return jsonify(prediction[0])

	except:
		pass

if __name__ == '__main__':
	app.run(host = '127.0.0.1',port=8080, debug =True)