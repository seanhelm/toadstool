"""
Supervised learning classification
"""
import pandas as pd
from timeit import default_timer as timer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class MultiClassifier:
    """
    Uses multiple statistical classification models to classify data

    :param data: Pandas DataFrame dataset
    :param models: Dictionary of classificiation models
    """
    def __init__(self, data, models):
        self.data = data
        self.models = models
        self.names = list(models.keys())        

    def preprocess(self, size=0.25):
        """
        Preprocesses the dataset for training
        """
        self.encoders = {}
        self.processed = {}

        for column in self.data.columns:
            # Encode all labels to value between 0..n-1
            self.encoders[column] = preprocessing.LabelEncoder()
            self.data[column] = self.encoders[column].fit_transform(self.data[column])
    
        # Split data into train and test
        data_X, data_y = self.data.iloc[:, 1:], self.data.iloc[:, 0]
        split_data = train_test_split(data_X, data_y, test_size=size)

        # Save y column name
        self.y_column = data_y.name
        
        self.processed['train_X'] = split_data[0]
        self.processed['test_X'] = split_data[1]
        self.processed['train_y'] = split_data[2]
        self.processed['test_y'] = split_data[3]

    def train_all(self):
        """
        Train the dataset on each statistical model
        """
        self.performances = []

        for key, model in self.models.items():
            start = timer()
            model.fit(self.processed['train_X'], self.processed['train_y'])
            end = timer()
            self.performances.append(end - start)

    def test_all(self):
        """
        Test the statistical models on the test data and output its accuracy
        
        :returns: Accuracy score for each model (0.0-1.0)
        """
        self.predictions = []
        self.accuracies = []

        for key, model in self.models.items():
            predict_y = model.predict(self.processed['test_X'])
            self.predictions.append(predict_y)
            accuracy = accuracy_score(self.processed['test_y'], predict_y)
            self.accuracies.append(accuracy)

    def predict_new(self, item):
        """
        Classify new item based on trained data

        :param item: Pandas DataFrame for prediction
        :returns: Dictionary of predictions from each model
        """
        for column in item.columns:
            item[column] = self.encoders[column].transform(item[column])
    
        # Predict using each model
        predictions = []
        for key, model in self.models.items():
            prediction = model.predict(item)
            # Decode back to actual label
            predictions.append(self.encoders[self.y_column].inverse_transform(prediction))

        return predictions