import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

from preprocess import preprocess

classifiers = {
    'Random Forest': RandomForestClassifier(),
    'Decision Tree Classifier': DecisionTreeClassifier(),
    'Logistic Regression': LogisticRegression(),
    'Gaussian Naive Bayes': GaussianNB()
}

def train(data, models):
    '''Train the dataset on each statistical model'''
    for key, model in models.items():
        model.fit(data['train_X'], data['train_y'])
        
def test(data, models):
    '''Test the statistical models on the test data and output its accuracy'''
    for key, model in models.items():
        predict_y = model.predict(data['test_X'])

        print("%s: %.2f" % (key, accuracy_score(data['test_y'], predict_y)))

if __name__ == '__main__':
    data_csv = pd.read_csv('../data/mushrooms.csv')
    data, encoder = preprocess(data_csv)

    train(data, classifiers)
    test(data, classifiers)
