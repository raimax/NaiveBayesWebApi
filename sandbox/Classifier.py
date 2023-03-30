import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


class Classifier:
    def __init__(self):
        self.accuracy = 0
        self.model = None
        self.vectorizer = None

    def initialize(self):
        data = pd.read_csv('https://raw.githubusercontent.com/AiDevNepal/ai-saturdays-workshop-8/master/data/spam.csv')
        data['target'] = np.where(data['target'] == 'spam', 1, 0)
        X_train, X_test, Y_train, Y_test = train_test_split(data['text'],
                                                            data['target'],
                                                            random_state=0)
        self.vectorizer = CountVectorizer(ngram_range=(1, 2)).fit(X_train)
        X_train_vectorized = self.vectorizer.transform(X_train)
        X_train_vectorized.toarray().shape
        self.model = MultinomialNB(alpha=0.1)
        self.model.fit(X_train_vectorized, Y_train)
        predictions = self.model.predict(self.vectorizer.transform(X_test))

        self.accuracy = 100 * sum(predictions == Y_test) / len(predictions)

    def predict(self, text):
        vectorizedText = self.vectorizer.transform([text])
        prediction = self.model.predict(vectorizedText)[0]
        return bool(prediction)

