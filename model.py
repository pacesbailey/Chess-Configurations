import keras_tuner as kt
import numpy as np

from keras import layers, losses, models, optimizers
from pathlib import Path
from preprocessing import encode_label, preprocess_image


def initialize_simple(hp: kt.BayesianOptimization) -> models.Sequential:
    """
    Defines and compiles a simple model architecture consisting of 
    convolutional neural network layers, max pooling layers, a dropout layer, 
    and a softmax layer.

    Args:
        hp (kt.BayesianOptimization): hyperparameters determined by a tuner

    Returns:
        models.Sequential: the initialized, hypertuned, untrained simple model
    """
    dropout_rate: float = hp.Choice(
        "dropout_rate", 
        values=[0.20, 0.35, 0.50, 0.65]
    )
    learning_rate: float = hp.Choice(
        "learning_rate", 
        values=[1e-1, 1e-2, 1e-3, 1e-4]
    )
    model: models.Sequential = models.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        layers.MaxPool2D(pool_size=(3, 3)),
        layers.Conv2D(filters=16, kernel_size=(5, 5), activation="relu"),
        layers.Flatten(),
        layers.Dropout(rate=dropout_rate),
        layers.Dense(units=13, activation="softmax")
    ])
    model.compile(
        optimizer=optimizers.legacy.Adam(learning_rate=learning_rate),
        loss=losses.CategoricalCrossentropy(),
        metrics=["accuracy"]
    )
    
    return model


def initialize_lenet5(hp: kt.BayesianOptimization) -> models.Sequential:
    """
    Defines and compiles an implementation of the LeNet-5, an early
    architecture proposed in 1998 for handwritten digit recognition.

    Args:
        hp (kt.BayesianOptimization): hyperparameters determined by a tuner

    Returns:
        models.Sequential: the initialized, hypertuned, untrained model
    """
    learning_rate: float = hp.Choice(
        "learning_rate",
        values=[1e-1, 1e-2, 1e-3, 1e-4]
    )
    model: models.Sequential = models.Sequential([
        layers.Conv2D(6, (5, 5), activation="relu", input_shape=(28, 28, 1)),
        layers.AveragePooling2D(),
        layers.Conv2D(16, (5, 5), activation="relu"),
        layers.AveragePooling2D(),
        layers.Flatten(),
        layers.Dense(120, activation="relu"),
        layers.Dense(84, activation="relu"),
        layers.Dense(13, activation="softmax")
    ])
    model.compile(
        optimizer=optimizers.legacy.Adam(learning_rate=learning_rate),
        loss=losses.CategoricalCrossentropy(),
        metrics=["accuracy"]
    )

    return model


def initialize_vgg(hp: kt.BayesianOptimization) -> models.Sequential:
    """
    Defines and compiles an implementation of the VGG (Visual Geometry 
    Group) model, which is a series of convolutional networks designed for use
    with facial recognition and image classification.

    Args:
        hp (kt.BayesianOptimization): hyperparameters determined by a tuner

    Returns:
        models.Sequential: the initialized, hypertuned, untrained model
    """
    learning_rate: float = hp.Choice(
        "learning_rate", 
        values=[1e-1, 1e-2, 1e-3, 1e-4]
    )
    model: models.Sequential = models.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPool2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPool2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(13, activation="softmax")
    ])
    model.compile(
        optimizer=optimizers.legacy.Adam(learning_rate=learning_rate),
        loss=losses.CategoricalCrossentropy(),
        metrics=["accuracy"]
    )

    return model


def generator(file_list: list, image_size: int) -> None:
    """
    A generator that yields the transformed images (x) and labels (y) for use 
    during the training and validation process.

    Args:
        file_list (list): list of filenames to iterate through
        image_size (int): size to which the images must be reduced
    """
    while True:
        for image in file_list:
            x: np.ndarray = preprocess_image(image, image_size)
            y: str = encode_label(image.stem)

            yield x, y


def test_generator(file_list: list, image_size: int) -> None:
    """
    A generator that yields the transformed images (x) and labels (y) for use 
    during the training and validation process.

    Args:
        file_list (list): list of filenames to iterate through
        image_size (int): size to which the images must be reduced
    """
    for image in file_list:
        x: np.ndarray = preprocess_image(image, image_size)
        y: str = encode_label(image.stem)

        yield x, y
    


def main() -> None:
    pass


if __name__ == "__main__":
    main()
