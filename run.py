"""
Demonstrate capabilities of learning library
"""
from __future__ import print_function

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

from toadstool.learn import MultiClassifier, train_percent_accuracy, feature_importances


if __name__ == '__main__':
    data_csv = pd.read_csv('data/mushrooms.csv')
    del data_csv['odor']
    del data_csv['gill-size']
    del data_csv['ring-type']
    del data_csv['gill-color']

    models = {
        'Random Forest': RandomForestClassifier(),
        'Decision Tree Classifier': DecisionTreeClassifier(),
        'Logistic Regression': LogisticRegression(),
        'Gaussian Naive Bayes': GaussianNB()
    }

    # Remove label column for predict_new
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