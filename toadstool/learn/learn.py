""""
Supervised learning classification
""""
import pandas as pd

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

__all__ = 'MultiClassifier'


class MultiClassifier:
    """
    Uses multiple statistical classification models to classify data

    :param data: Pandas DataFrame dataset
    :param models: Dictionary of classificiation models
    """
    def __init__(self, data, models):
        self.data = data
        self.models = models
        self.encoders = {}
        self.processed = {}

    def preprocess(self):
        """
        Preprocesses the dataset for training
        """
        for column in self.data.columns:
            # Encode all labels to value between 0..n-1
            self.encoders[column] = preprocessing.LabelEncoder()
            self.data[column] = self.encoders[column].fit_transform(self.data[column])
    
        # Split data into train and test
        data_X, data_y = self.data.iloc[:, 1:], self.data.iloc[:, 0]
        split_data = train_test_split(data_X, data_y)
        
        self.processed['train_X'] = split_data[0]
        self.processed['test_X'] = split_data[1]
        self.processed['train_y'] = split_data[2]
        self.processed['test_y'] = split_data[3]

    def train_all(self):
        """
        Train the dataset on each statistical model
        """
        for model in self.models.values():
            model.fit(self.processed['train_X'], self.processed['train_y'])

    def test_all(self):
        """
        Test the statistical models on the test data and output its accuracy
        
        :returns: Accuracy score for each model (0.0-1.0)
        """
        accuracy = {}

        for key, model in self.models.items():
            predict_y = model.predict(self.processed['test_X'])
            accuracy[key] = accuracy_score(self.processed['test_y'], predict_y)

        return accuracy

    def predict_new(self, item):
        """
        Classify new item based on trained data

        :param item: Pandas DataFrame for prediction
        :returns: Dictionary of predictions from each model
        """
        for column in item.columns:
            item[column] = self.encoders[column].transform(item[column])
    
        # Predict using each model
        predictions = {}
        for key, model in self.models.items():
            predictions[key] = model.predict(item)

        return predictions
