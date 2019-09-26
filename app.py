import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from sklearn.externals import joblib
from sklearn.naive_bayes import GaussianNB


app = Flask(__name__)

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/train')
def train():
	df_train = pd.read_excel('False Alarm Cases.xlsx')
	df_train = df_train.iloc[:, 1:8]
	X = df_train.iloc[:,0:6]
	y = df_train['Spuriosity Index(0/1)']

	classifier = GaussianNB()
	classifier.fit(X, y)
	joblib.dump(classifier, 'filename.pkl')

	return jsonify({'message':'Model has been Trained.'})

@app.route('/test', methods=['POST'])
def test():

	columnslf = joblib.load('filename.pkl')

	request_data = request.get_json()

	a = request_data['Ambient Temperature']
	b = request_data['Calibration']
	c = request_data['Unwanted substance deposition']
	d = request_data['Humidity']
	e = request_data['H2S Content']
	f = request_data['detected by']
	l = [a,b,c,d,e,f]
	narr = np.array(l)
	narr = narr.reshape(1,6)
	df_test = pd.DataFrame(narr, columns = ['Ambient Temperature', 'Calibration', 'Unwanted substance deposition',
	'Humidity', 'H2S Content', 'detected by'])

	ypred = columnslf.predict(df_test)

	if ypred ==1:
		result = 'Danger'

	else:
		result='No Danger'

	return jsonify({'Recommendation':result})


@app.route('/testing',methods=['POST'])
def testing():
	columnslf = joblib.load('filename.pkl')


	a = int(request.form['AmbientTemperature'])
	b = int(request.form['Calibration'])
	c = int(request.form['Unwantedsubstancedeposition'])
	d = int(request.form['Humidity'])
	e = int(request.form['H2SContent'])
	f = int(request.form['detectedby'])
	l = [a, b, c, d, e, f]

	narr = np.array(l)
	narr = narr.reshape(1, 6)
	df_test = pd.DataFrame(narr, columns=['Ambient Temperature', 'Calibration', 'Unwanted substance deposition',
										  'Humidity', 'H2S Content', 'detected by'])

	ypred = columnslf.predict(df_test)

	if ypred == 1:
		result = 'Danger'

	else:
		result = 'No Danger'

	return jsonify({'Recommendation': result})



@app.route('/testing2',methods=['POST','GET'])
def testing2():
	return jsonify({'Recommendation':str(request.form)})

app.run(port=4000,debug=False)