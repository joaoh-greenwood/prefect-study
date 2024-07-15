from prefect import task, flow, get_run_logger, serve
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

@flow(name='churn-prediction-flow')
def churn_prediction_flow():
    logger = get_run_logger()

    # Importing Data
    df_state = import_data.submit()
    logger.info(f"SUBMIT: Importing Data Task Status: {df_state}")
    df = df_state.result()
    logger.info(f"RESULT: Importing Data Task Status: {df_state}")

    # Pre-Process
    preprocess_state = preprocess_data.submit(df)
    logger.info(f"SUBMIT: Pre-Process Data Task Status: {preprocess_state}")
    X, y = preprocess_state.result()
    logger.info(f"RESULT: Pre-Process Data Task Status: {preprocess_state}")

    # Split Data
    split_state = split_data.submit(X, y)
    logger.info(f"SUBMIT: Split Data Task Status: {split_state}")
    X_train, X_test, y_train, y_test = split_state.result()
    logger.info(f"RESULT: Split Data Task Status: {split_state}")

    # Build Model
    build_state = build_model.submit(input_dim=X_train.shape[1])
    logger.info(f"SUBMIT: Build Model Task Status: {build_state}")
    model = build_state.result()
    logger.info(f"RESULT: Build Model Task Status: {build_state}")

    # Train Model
    train_state = train_model.submit(model, X_train, y_train)
    logger.info(f"SUBMIT: Train Model Task Status: {train_state}")
    trained_model = train_state.result()
    logger.info(f"RESULT: Train Model Task Status: {train_state}")

    # Evaluate Model
    evaluate_state = evaluate_model.submit(trained_model, X_test, y_test)
    logger.info(f"SUBMIT: Evaluate Model Task Status: {evaluate_state}")
    accuracy, f1, recall, conf_matrix = evaluate_state.result()
    logger.info(f"RESULT: Evaluate Model Task Status: {evaluate_state}")

    print_evaluation(accuracy, f1, recall, conf_matrix)

if __name__ == "__main__":
    deployment = churn_prediction_flow.to_deployment(name="rna-analisys")
    deployment.apply()
    
    serve(deployment)
