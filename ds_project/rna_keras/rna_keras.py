from prefect import task, flow
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Dropout
import os

@task
def import_data():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    df = pd.read_csv("./churn.csv", sep=";")
    return df

@task
def preprocess_data(df):
    X = df.drop("Exited", axis=1)
    y = df["Exited"]

    # Standardize numerical features
    standardScaler = StandardScaler()
    numerical = X.select_dtypes(include=['int64', 'float64']).columns
    X[numerical] = standardScaler.fit_transform(X[numerical])

    # Encode categorical features
    labelencoder = LabelEncoder()
    categorical = X.select_dtypes(include='object').columns
    for col in categorical:
        X[col] = labelencoder.fit_transform(X[col])

    return X, y

@task
def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    return X_train, X_test, y_train, y_test

@task
def build_model(input_dim):
    model = Sequential()
    model.add(Dense(units=64, activation='relu', input_dim=input_dim))
    model.add(Dropout(0.4))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

@task
def train_model(model, X_train, y_train):
    model.fit(X_train, y_train, epochs=50, batch_size=32)
    return model

@task
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    y_pred = (predictions > 0.5).astype('int32')

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    return accuracy, f1, recall, conf_matrix

@task
def print_evaluation(accuracy, f1, recall, conf_matrix):
    print('Accuracy: ', accuracy)
    print('F1: ', f1)
    print('Recall: ', recall)
    print('Conf Matrix: ', conf_matrix)

@flow(name='Churn Prediction Flow')
def churn_prediction_flow():
    df = import_data()
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = build_model(input_dim=X_train.shape[1])
    trained_model = train_model(model, X_train, y_train)
    accuracy, f1, recall, conf_matrix = evaluate_model(trained_model, X_test, y_test)
    print_evaluation(accuracy, f1, recall, conf_matrix)

if __name__ == "__main__":
    churn_prediction_flow()
