from flask import Flask, jsonify
from flask_cors import CORS

from Classifier import Classifier
from Algorithm import Algorithm

app = Flask(__name__)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})

naivePredictor = Classifier(Algorithm.NaiveBayes)
naivePredictor.initialize()

logisticPredictor = Classifier(Algorithm.LogisticRegression)
logisticPredictor.initialize()


@app.route('/api/naive-bayes/predict/<message>', methods=['GET'])
def naivePredict(message):
    return jsonify(isSpam=naivePredictor.predict(message))


@app.route('/api/naive-bayes/model', methods=['GET'])
def getNaiveModel():
    return jsonify(accuracy=naivePredictor.accuracy)


@app.route('/api/logistic-regression/predict/<message>', methods=['GET'])
def logisticPredict(message):
    return jsonify(isSpam=logisticPredictor.predict(message))


@app.route('/api/logistic-regression/model', methods=['GET'])
def getLogisticModel():
    return jsonify(accuracy=logisticPredictor.accuracy)
