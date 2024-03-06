import numpy as np

from keras import layers, models
from pathlib import Path
from preprocessing import preprocess_image, encode_label


def initialize_simple(shape: tuple) -> models.Sequential:
    """
    Defines and compiles a simple model architecture consisting of 
    convolutional neural network layers, max pooling layers, a dropout layer, 
    and a softmax layer.

    Args:
        shape (tuple): input shape given as a tuple

    Returns:
        models.Sequential: the initialized, untrained simple model
    """
    model: models.Sequential = models.Sequential(
        [
            layers.Conv2D(
                filters=32, kernel_size=(3, 3), 
                activation="relu", input_shape=shape
            ),
            layers.MaxPool2D(pool_size=(3, 3)),
            layers.Conv2D(
                filters=16, kernel_size=(5, 5), 
                activation="relu"
            ),
            layers.Flatten(),
            layers.Dropout(rate=0.35),
            layers.Dense(units=13, activation="softmax")
        ]
    )
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model


def initialize_vgg(shape: tuple) -> models.Sequential:
    """
    Defines and compiles an implementation of the VGG-16 (Visual Geometry 
    Group) model, which is a series of convolutional networks designed for use
    with facial recognition and image classification.

    Args:
        shape (tuple): input shape given as a tuple

    Returns:
        models.Sequential: initialized, untrained VGG-16 model
    """
    model: models.Sequential = models.Sequential(
        [
            layers.Conv2D(
                filters=64, kernel_size=(3,3), 
                padding="same", activation="relu", 
                input_shape=shape
            ),
            layers.Conv2D(
                filters=64, kernel_size=(3,3), 
                padding="same", activation="relu"
            ),
            layers.MaxPool2D(
                pool_size=(2, 2), strides=(2, 2), 
                padding="same"
            ),
            layers.Conv2D(
                filters=128, kernel_size=(3,3), 
                padding="same", activation="relu"
            ),
            layers.Conv2D(
                filters=128, kernel_size=(3,3), 
                padding="same", activation="relu"
            ),
            layers.MaxPool2D(
                pool_size=(2, 2), strides=(2, 2), 
                padding="same"
            ),
            layers.Conv2D(
                filters=256, kernel_size=(3,3), 
                padding="same", activation="relu"
            ),
            layers.Conv2D(
                filters=256, kernel_size=(3,3), 
                padding="same", activation="relu"
            ),
            layers.Conv2D(
                filters=256, kernel_size=(3,3), 
                padding="same", activation="relu"
            ),
            layers.MaxPool2D(
                pool_size=(2, 2), strides=(2, 2), 
                padding="same"
            ),
            layers.Conv2D(
                filters=512, kernel_size=(3,3), 
                padding="same", activation="relu"
            ),
            layers.Conv2D(
                filters=512, kernel_size=(3,3), 
                padding="same", activation="relu"
            ),
            layers.Conv2D(
                filters=512, kernel_size=(3,3), 
                padding="same", activation="relu"
            ),
            layers.MaxPool2D(
                pool_size=(2, 2), strides=(2, 2), 
                padding="same"
            ),
            layers.Conv2D(
                filters=512, kernel_size=(3,3), 
                padding="same", activation="relu"
            ),
            layers.Conv2D(
                filters=512, kernel_size=(3,3), 
                padding="same", activation="relu"
            ),
            layers.Conv2D(
                filters=512, kernel_size=(3,3), 
                padding="same", activation="relu"
            ),
            layers.MaxPool2D(
                pool_size=(2, 2), strides=(2, 2), 
                padding="same"
            ),
            layers.Flatten(),
            layers.Dense(
                units=4096, 
                activation="relu"
            ),
            layers.Dense(
                units=4096, 
                activation="relu"
            ),
            layers.Dense(
                units=13, 
                activation="softmax"
            )
        ]
    )
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
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
    for image in file_list:
        x: np.ndarray = preprocess_image(image, image_size)
        y: str = encode_label(image.stem)

        yield x, y


def main() -> None:
    pass


if __name__ == "__main__":
    main()
