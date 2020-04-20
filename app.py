from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib



app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')
@app.route ('/demo',methods=['GET','POST'])
def demo():
	return render_template('demo.html')

@app.route('/predict',methods=['POST'])
def predict():
	df= pd.read_csv("dataset.csv")
	df_data = df[["CONTENT","CLASS"]]
	# Features and Labels
	x = df_data['CONTENT']
	y = df_data.CLASS
    # Extract Feature With CountVectorizer
	corpus = x
	cv = CountVectorizer()
	X = cv.fit_transform(corpus) # Fit the Data
	from sklearn.model_selection import train_test_split
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
	#Naive Bayes Classifier
	from sklearn.naive_bayes import MultinomialNB
	clf = MultinomialNB(alpha=0.2)
	clf.fit(X_train,y_train)
	if request.method == 'POST':
		comment = request.form['comment']
		data = [comment]
		if len(comment)<=0:
			return render_template('nodata.html')
		else:
			vect = cv.transform(data).toarray()
			my_prediction = clf.predict(vect)
	return render_template('result.html',prediction = my_prediction)

if __name__ == '__main__':
	app.run(debug=True)