import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

data = pd.read_csv('data/raw/mushrooms.csv')
encoder = preprocessing.LabelEncoder()

for column in data.columns:
    data[column] = encoder.fit_transform(data[column])

data_X, data_y = data.iloc[:, 1:], data.iloc[:, 0]
X_train, X_test, y_train, y_test = train_test_split(data_X, data_y)

models = {
    'Random Forest': RandomForestClassifier(),
    'Decision Tree Classifier': DecisionTreeClassifier(),
    'Logistic Regression': LogisticRegression(),
    'Gaussian Naive Bayes': GaussianNB()
}

for key, model in models.items():
    model.fit(X_train, y_train)
    predict_y = model.predict(X_test)

    print("%s: %.2f" % (key, accuracy_score(y_test, predict_y)))
