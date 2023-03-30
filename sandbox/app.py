from flask import Flask, jsonify

from Classifier import Classifier
from Algorithm import Algorithm

app = Flask(__name__)

naivePredictor = Classifier(Algorithm.NaiveBayes)
naivePredictor.initialize()

logisticPredictor = Classifier(Algorithm.LogisticRegression)
logisticPredictor.initialize()


@app.route('/naive-bayes/predict/<message>', methods=['GET'])
def naivePredict(message):
    return jsonify(isSpam=naivePredictor.predict(message))


@app.route('/naive-bayes/model', methods=['GET'])
def getNaiveModel():
    return jsonify(accuracy=naivePredictor.accuracy)


@app.route('/logistic-regression/predict/<message>', methods=['GET'])
def logisticPredict(message):
    return jsonify(isSpam=logisticPredictor.predict(message))


@app.route('/logistic-regression/model', methods=['GET'])
def getLogisticModel():
    return jsonify(accuracy=logisticPredictor.accuracy)
