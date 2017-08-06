"""
Plot classification models in different ways to analyze quality
"""
import numpy as np
import matplotlib.pyplot as plt

from learn import MultiClassifier

__all__ = ['train_percent_accuracy']

def train_percent_accuracy(multi_classifier):
    """
    Plot training percentange vs. accuracy score from 1% to 99%

    :param multi_classifier: MultiClassifier instance to plot
    """
    plots = {k: ([],[]) for (k, v) in multi_classifier.models.items()}

    for i in np.arange(0.01, 1, 0.01):
        multi_classifier.preprocess(1.0-i)
        multi_classifier.train_all()
        test_results = multi_classifier.test_all()

        for model, result in test_results.items():
            plots[model][0].append(i)
            plots[model][1].append(result)

    for model, results in plots.items():
        plt.plot(results[0], results[1], label=model)
    
    plt.xlabel("Training %")
    plt.ylabel("Accuracy score")
    plt.legend()
    plt.show()