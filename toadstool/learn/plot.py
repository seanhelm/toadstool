"""
Plot classification models in different ways to analyze quality
"""
import numpy as np
import matplotlib.pyplot as plt

from .learn import MultiClassifier


def train_percent_accuracy(multi_classifier, step=0.01):
    """
    Plot training percentange vs. accuracy score from 1% to 99%

    :param multi_classifier: MultiClassifier instance to plot
    """
    plots = {k: ([],[]) for (k, v) in multi_classifier.models.items()}

    # Iterate through all training percentages
    for i in np.arange(0.01, 1, step):
        multi_classifier.preprocess(1.0-i)
        multi_classifier.train_all()
        multi_classifier.test_all()

        # Plot each model's accuracy score for this training percentage
        for j in range(len(multi_classifier.accuracies)):
            plots[multi_classifier.names[j]][0].append(i)
            plots[multi_classifier.names[j]][1].append(multi_classifier.accuracies[j])

    plt.figure()
    for model, results in plots.items():
        plt.plot(results[0], results[1], label=model)
    
    plt.xlabel("Dataset Training %")
    plt.ylabel("Accuracy score")
    plt.legend()
    plt.show()

def feature_importances(feature_importances, features):
    """
    Plot feature importances on a bar graph

    :param feature_importances: List of feature importances [0.0-1.0]
    :param features: List of dataset features
    """
    plt.figure()
    x_pos = np.arange(len(features))
    
    plt.bar(x_pos, feature_importances, align='center')
    plt.gcf().subplots_adjust(bottom=0.3)
    plt.xticks(x_pos, features, rotation=45, rotation_mode='anchor', ha='right')
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.show()

def performance_all(performances, model_names):
    """
    Plot each model's performance when training on the dataset

    :param performances: Dictionary with performance in seconds per model
    """
    plt.figure()

    x_pos = np.arange(len(performances))

    plt.bar(x_pos, performances, align='center', width=0.4)
    plt.gcf().subplots_adjust(bottom=0.35)
    plt.xticks(x_pos, model_names, rotation=45, rotation_mode='anchor', ha='right')
    plt.xlabel("Model")
    plt.ylabel("Performance (s)")
    plt.show()