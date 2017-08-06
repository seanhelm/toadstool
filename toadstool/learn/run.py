"""
Demonstrate capabilities of learning library
"""
from __future__ import print_function

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

from learn import MultiClassifier


if __name__ == '__main__':
    data_csv = pd.read_csv('../data/mushrooms.csv')

    models = {
        'Random Forest': RandomForestClassifier(),
        'Decision Tree Classifier': DecisionTreeClassifier(),
        'Logistic Regression': LogisticRegression(),
        'Gaussian Naive Bayes': GaussianNB()
    }

    data_instance = data_csv.loc[[5000]]
    del data_instance['class']

    classifier = MultiClassifier(data_csv, models)
    
    classifier.preprocess()
    classifier.train_all()

    test_results = classifier.test_all()
    print("Accuracy score for all models:")
    for model, result in test_results.items():
        print("%s: %.3f" % (model, result))
    print()

    predictions= classifier.predict_new(data_instance)
    print("Prediction for new mushroom data:")
    for model, prediction in predictions.items():
        print("%s: %s" % (model, prediction[0]))
    print()