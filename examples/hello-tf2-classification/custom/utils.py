import pandas as pd
import sklearn.datasets as dt
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_cancer_dataset(n_samples=100000, test_size=0.25):
    """
    Making a simulated dataset to mirror my real use case
    """
    # running the statement below is going to be slow everytime but works to
    # show just the sample set
    print("Creating the dataset t his takes a little while ...")
    X, Y = dt.make_classification(n_samples=n_samples,
                                  n_classes=10,
                                  n_features=2000,
                                  n_repeated=100,
                                  n_redundant=300,
                                  n_informative=300,
                                  random_state=10)

    print("Done.")
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)
    y = np_utils.to_categorical(encoded_Y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    print("Size of train dataset: {} rows".format(X_train.shape[0]))
    print("Size of test dataset: {} rows".format(X_test.shape[0]))
    return X_train, X_test, y_train, y_test


def get_simple_sequential_model(feature_dim):
    model = Sequential()
    model.add(Dense(100, input_dim=feature_dim, name='layer_1', activation='relu', use_bias=False))
    model.add(Dense(10, name='layer_2', activation='softmax', use_bias=False))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
