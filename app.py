from flask import Flask, request, jsonify, send_file
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, roc_curve, auc
from flask_cors import CORS
import re
from customtokenizer import tokenizer, tokenizer_porter


app = Flask(__name__)
CORS(app,resources={r"/*": {"origins": "*"}})

# Load the models and transformers
model_lr = joblib.load('./models/lr_model.pkl')
model_dt = joblib.load('./models/dt_model.pkl')
model_rf = joblib.load('./models/rf_model.pkl')
model_nb = joblib.load('./models/nb_model.pkl')
tfidf = joblib.load('./models/tfidf_vectorizer.pkl')

def process_tweet(tweet):
    return " ".join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])", " ", tweet.lower()).split())

def predict_new_tweet(tweet, model):
    processed_tweet = process_tweet(tweet)
    tweet_tfidf = tfidf.transform([processed_tweet])
    prediction = model.predict(tweet_tfidf)
    pred_prob = model.predict_proba(tweet_tfidf)
    return prediction, pred_prob

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    tweet = data['tweet']
    model_name = data.get("model", "lr")  # default to logistic regression
    if model_name == 'lr':
        prediction, pred_prob = predict_new_tweet(tweet, model_lr)
    elif model_name == 'dt':
        prediction, pred_prob = predict_new_tweet(tweet, model_dt)
    elif model_name == 'rf':
        prediction, pred_prob = predict_new_tweet(tweet, model_rf)
    elif model_name == 'nb':
        prediction, pred_prob = predict_new_tweet(tweet, model_nb)
    else:
        return jsonify({'error': 'Invalid model name'}), 400

    pred_label = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}.get(prediction[0], "UNKNOWN")
    listt = pred_prob.tolist()
    for i in range(3):
        listt[0][i] *= 100
        listt[0][i] = round(listt[0][i], 2)
    return jsonify({'prediction': pred_label, 'probabilities': [f' Negative : {listt[0][0]} %',f' Neutral : {listt[0][1]} %',f' Positive : {listt[0][2]} %'] })

@app.route('/accuracy', methods=['POST'])
def getAccuracy():
    data = request.get_json(force=True)
    model_name = data.get("model", "lr")  # default to logistic regression
    if model_name == 'lr':
        accuracy = "68%"
    elif model_name == 'dt':
        accuracy = "65%"
    elif model_name == 'rf':
        accuracy = "70%"
    elif model_name == 'nb':
        accuracy = "63%"
    else:
        return jsonify({'error': 'Invalid model name'}), 400
    
    return jsonify({"accuracy":accuracy })

@app.route('/roc_curve', methods=['GET'])
def get_roc_curve():
    model_name = request.args.get('model', 'lr')  # default to logistic regression
    if model_name not in ['lr', 'dt', 'rf', 'nb']:
        return jsonify({'error': 'Invalid model name'}), 400

    # Assuming the ROC curve images are saved with names corresponding to the models
    image_path = f"{model_name}.jpg"

    try:
        return jsonify({'image_path': image_path})
    except FileNotFoundError:
        return jsonify({'error': 'Image not found'}), 404

@app.route('/classification_report', methods=['GET'])
def get_report():
    model_name = request.args.get('model', 'lr')  # default to logistic regression
    if model_name not in ['lr', 'dt', 'rf', 'nb']:
        return jsonify({'error': 'Invalid model name'}), 400

    # Assuming the ROC curve images are saved with names corresponding to the models
    file_path = f"{model_name}_classification_report.txt"

    try:
        return jsonify({'file_path': file_path})
    except FileNotFoundError:
        return jsonify({'error': 'Image not found'}), 404

if __name__ == '__main__':
    app.run(debug=True)
