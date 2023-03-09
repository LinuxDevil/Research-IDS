from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np
from sklearn.model_selection import train_test_split


def build_cnn(input_shape, num_classes):
    """
    Builds a CNN model for intrusion detection.

    Parameters:
    input_shape (tuple): The shape of the input data.
    num_classes (int): The number of classes in the output.

    Returns:
    keras.models.Sequential: The compiled CNN model.
    """
    # Define the model architecture
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dense(num_classes, activation="softmax"))

    # Compile the model
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model


if __name__ == '__main__':
    # Load the data
    data = np.load("intrusion_data.npy")
    labels = np.load("intrusion_labels.npy")

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    # Normalize the data
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0

    # Convert the labels to one-hot encoded vectors
    num_classes = len(np.unique(labels))
    y_train = np.eye(num_classes)[y_train]
    y_test = np.eye(num_classes)[y_test]

    # Build the CNN model
    input_shape = (32, 32, 3)  # example shape
    num_classes = 2  # example number of classes
    model = build_cnn(input_shape, num_classes)

    # Train the model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)
