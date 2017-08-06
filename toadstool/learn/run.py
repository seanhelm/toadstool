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
    print(classifier.test_all())

    prediction = classifier.predict_new(data_instance)
    print(prediction)