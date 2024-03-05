import numpy as np

from keras import layers, models
from pathlib import Path
from preprocessing import transform_image, transform_label


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
        percentage (float): 
    """
    directory: list = sorted(directory.iterdir())

    for image in directory[:int(len(directory) * percentage)]:
        x: np.ndarray = transform_image(image, image_size)
        y: str = transform_label(image.stem)

        yield x, y
