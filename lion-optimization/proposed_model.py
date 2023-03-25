import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from hyper_param_loa import loa

# traffic analyzer->
# preprocessing->
# Feature selection ->
# 1. Training 2.Testing ->
# Intrusion detection classifier(CNN) ->
# alert


def traffic_analyzer(raw_data):
    # Parse the raw data
    data = pd.read_csv(raw_data, header=None)

    # Combine the train and test data
    data = pd.concat([data.iloc[:, :-2], data.iloc[:, -1]], axis=1)

    # Rename the columns
    data.columns = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment',
                    'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
                    'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
                    'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
                    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
                    'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
                    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
                    'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label']

    # Drop irrelevant columns
    data = data.drop(['duration', 'src_bytes', 'dst_bytes', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
                      'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
                      'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login'], axis=1)

    # Encode the labels using a LabelEncoder
    le = LabelEncoder()
    data.iloc[:, -1] = le.fit_transform(data.iloc[:, -1])

    # Split the data into features and labels
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


def preprocess_data(X_train, X_test):
    # Reshape the data into a 4D tensor
    X_train = X_train.reshape(-1, 1, X_train.shape[1], 1)
    X_test = X_test.reshape(-1, 1, X_test.shape[1], 1)

    return X_train, X_test


def select_features(X_train, X_test):
    # Select the features to use
    selected_features = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 15, 22]

    # Filter the features
    X_train = X_train[:, selected_features]
    X_test = X_test[:, selected_features]

    return X_train, X_test


def train_and_test(raw_data):
    # Analyze the traffic
    X_train, X_test, y_train, y_test = traffic_analyzer(raw_data)

    # Preprocess the data
    X_train, X_test = preprocess_data(X_train, X_test)

    # Select the features to use
    X_train, X_test = select_features(X_train, X_test)

    # Define the CNN model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(1, 14, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model using LOA
    best_params, best_score = loa(model.fit, {"X": X_train, "y": y_train, "batch_size": 128, "epochs": 10, "verbose": 0},
                                  max_iterations=50, num_agents=10, lower_bound=0.0001, upper_bound=0.1,
                                  dimension=4)

    # Print the best hyperparameters and their score
    print("Best hyperparameters:", best_params)
    print("Best score:", best_score)

    # Evaluate the model on the test data
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print("Test accuracy:", accuracy)

    # Generate alerts for the test data
    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5)
    print("Number of alerts:", np.sum(y_pred))


def intrusion_detection_classifier(X_train, y_train, X_test, y_test):
    # Reshape the data into a 4D tensor
    X_train = X_train.reshape(-1, 1, X_train.shape[1], 1)
    X_test = X_test.reshape(-1, 1, X_test.shape[1], 1)

    # Encode the labels using a LabelEncoder
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    # Define the CNN model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(1, 14, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, batch_size=128, epochs=10, verbose=0)

    # Evaluate the model on the test data
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print("Test accuracy:", accuracy)

    # Generate alerts for the test data
    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5)
    print("Number of alerts:", np.sum(y_pred))

    return model


def generate_alerts(model, X):
    # Reshape the data into a 4D tensor
    X = X.reshape(-1, 1, X.shape[1], 1)

    # Generate predictions using the trained model
    y_pred = model.predict(X)
    y_pred = (y_pred > 0.5)

    # Identify the indexes of the alert samples
    alert_indexes = np.where(y_pred == True)[0]

    # Print the number of alerts
    print("Number of alerts:", len(alert_indexes))

    # Return the indexes of the alert samples
    return alert_indexes
