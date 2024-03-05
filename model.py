import numpy as np

from keras import layers, models
from pathlib import Path
from preprocessing import preprocess_image, encode_label


def initialize_vgg(shape: tuple) -> models.Sequential:
    """
    Description

    Args:
        shape (tuple):

    Returns:
        models.Sequential:
    """
    model = models.Sequential(
        [
            layers.Conv2D(64, (3,3), padding="same", activation="relu", input_shape=shape),
            layers.Conv2D(64, (3,3), padding="same", activation="relu"),
            layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same"),
            layers.Conv2D(128, (3,3), padding="same", activation="relu"),
            layers.Conv2D(128, (3,3), padding="same", activation="relu"),
            layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same"),
            layers.Conv2D(256, (3,3), padding="same", activation="relu"),
            layers.Conv2D(256, (3,3), padding="same", activation="relu"),
            layers.Conv2D(256, (3,3), padding="same", activation="relu"),
            layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same"),
            layers.Conv2D(512, (3,3), padding="same", activation="relu"),
            layers.Conv2D(512, (3,3), padding="same", activation="relu"),
            layers.Conv2D(512, (3,3), padding="same", activation="relu"),
            layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same"),
            layers.Conv2D(512, (3,3), padding="same", activation="relu"),
            layers.Conv2D(512, (3,3), padding="same", activation="relu"),
            layers.Conv2D(512, (3,3), padding="same", activation="relu"),
            layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same"),
            layers.Flatten(),
            layers.Dense(4096, activation="relu"),
            layers.Dense(4096, activation="relu"),
            layers.Dense(13, activation="softmax")
        ]
    )

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def initialize_simple(shape: tuple) -> models.Sequential:
    """
    Defines and compiles a simple model architecture consisting of 
    convolutional neural network layers, max pooling layers, a dropout layer, 
    and a softmax layer.

    Args:
        shape (tuple): contains the input shape

    Returns:
        models.Sequential: the initialized, untrained simple model
    """
    model = models.Sequential(
        [
            layers.Conv2D(32, (3, 3), activation="relu", input_shape=shape),
            layers.MaxPool2D(pool_size=(3, 3)),
            layers.Conv2D(16, (5, 5), activation="relu"),
            layers.Flatten(),
            layers.Dropout(0.35),
            layers.Dense(13, activation="softmax")
        ]
    )

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def generator(directory: Path, image_size: int, percentage: float) -> None:
    """
    A generator that yields the transformed images (x) and labels (y) for use 
    during the training and validation process.

    Args:
        directory (Path): path object leading to directory with contains 
                          the training or validation data
        image_size (int): size to which the images must be reduced
        percentage (float): limits the amount of the dataset to be used
    """
    directory: list = sorted(directory.iterdir())

    for image in directory[:int(len(directory) * percentage)]:
        x: np.ndarray = preprocess_image(image, image_size)
        y: str = encode_label(image.stem)

        yield x, y


def main() -> None:
    pass


if __name__ == "__main__":
    main()
