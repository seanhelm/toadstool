"""
Demonstrate capabilities of learning library
"""
from __future__ import print_function

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

from toadstool.learn import *


if __name__ == '__main__':
    data_csv = pd.read_csv('data/mushrooms.csv')

    models = {
        'Random Forest': RandomForestClassifier(),
        'Decision Tree Classifier': DecisionTreeClassifier(),
        'Logistic Regression': LogisticRegression(),
        'Gaussian Naive Bayes': GaussianNB()
    }

    # Remove label column for predict_new (not new but done to show functionality)
    data_instance = data_csv.loc[[5000]]
    del data_instance['class']

    # Initialize new multi-classification object
    classifier = MultiClassifier(data_csv, models)
    
    # Preprocess data and train
    classifier.preprocess()
    classifier.train_all()

    # Display the accuracy scores for each model
    test_results = classifier.test_all()
    print("Accuracy score for all models:")
    for model, result in test_results.items():
        print("%s: %.3f" % (model, result))
    print()

    # Visualize the performance of each model on this dataset
    performance_all(classifier.performances)
    '''
    # Make prediction on new data
    predictions= classifier.predict_new(data_instance)
    print("Prediction for new mushroom data:")
    for model, prediction in predictions.items():
        print("%s: %s" % (model, prediction[0]))
    print()

    # Visualize feature importance
    importances = classifier.models['Random Forest'].feature_importances_
    feature_columns = classifier.processed['train_X'].columns
    feature_importances(importances, feature_columns)

    # Visualize relationship between classification quality and training percentage
    train_percent_accuracy(classifier)
    '''