from flask import Flask, jsonify

from Classifier import Classifier

app = Flask(__name__)

predictor = Classifier()
predictor.initialize()


@app.route('/<message>', methods=['GET'])
def index(message):
    return jsonify(isSpam=predictor.predict(message))


@app.route('/model', methods=['GET'])
def getModel():
    return jsonify(accuracy=predictor.accuracy)
