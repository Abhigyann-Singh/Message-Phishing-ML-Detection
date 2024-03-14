from flask import Flask ,render_template, request
import pickle
import numpy as np

app = Flask(__name__)
vectorizer = pickle.load(open('Mlmodels/tfidf.pkl', 'rb'))
model = pickle.load(open('Mlmodels/model.pkl', 'rb'))
@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/message')
def products():
    return render_template('messages.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        # print(request.form)
        message = request.form['message']
        # print(message)
        message = np.array([message])
        ttransformed = vectorizer.transform(message)
        my_prediction = model.predict(ttransformed)
    if my_prediction[0] == 0:
        return render_template('messages.html',msggg= message[0], prediction='Not Spam')
    else:
        return render_template('messages.html',msggg= message[0], prediction='Spam')
    

if __name__ == '__main__':
    app.run(debug=True)