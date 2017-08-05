from sklearn import preprocessing
from sklearn.model_selection import train_test_split

def preprocess(data):
    '''Preprocesses the dataset for training'''
    encoder = preprocessing.LabelEncoder()

    for column in data.columns:
        data[column] = encoder.fit_transform(data[column])
    
    data_X, data_y = data.iloc[:, 1:], data.iloc[:, 0]

    split_data = train_test_split(data_X, data_y)

    processed = {
        'train_X': split_data[0],
        'test_X': split_data[1],
        'train_y': split_data[2],
        'test_y': split_data[3]
    }

    return processed, encoder