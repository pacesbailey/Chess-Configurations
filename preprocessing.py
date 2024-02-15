import numpy as np
import random
import re

from pathlib import Path
from skimage import io
from skimage.util.shape import view_as_blocks


def preprocess_dataset(train_directory: Path, test_directory: Path) -> tuple:
    """
    Performs several preprocessing steps to the dataset, including loading the
    images into arrays with their individual cells separated and transforming
    the FEN labels of the images into one-hot encodings, then returns the
    training and test data in a tuple, separated by images and labels.

    Args:
        train_directory (Path): leads to the train subfolder
        test_directory (Path): leads to the test subfolder

    Returns:
        tuple: contains the preprocessed train and test datasets
    """
    train_files: list = list(train_directory.iterdir())
    test_files: list = list(test_directory.iterdir())
    
    random.shuffle(train_files)
    random.shuffle(test_files)

    x_train, y_train, x_test, y_test = split_dataset(train_files, test_files)

    x_train = [transform_image(image) for image in x_train]
    x_test = [transform_image(image) for image in x_test]
    y_train = [transform_label(label) for label in y_train]
    y_test = [transform_label(label) for label in y_test]

    return (x_train, y_train, x_test, y_test)


def split_dataset(train_list: list[Path], test_list: list[Path]) -> tuple:
    """
    Applies transformations to the images, splits their FEN labels off into a
    separate list, then returns a tuple with separate images (x) and labels
    (y) for both the training and test datasets.

    Args:
        train_list (list[Path]): contains paths to the training dataset
        test_list (list[Path]): contains paths to the test dataset

    Returns:
        tuple: contains images and FEN labels for training and test datasets
    """
    x_train: list = train_list
    x_test: list = test_list

    y_train: list[str] = [file.stem for file in train_list]
    y_test: list[str] = [file.stem for file in test_list]

    return (x_train, y_train, x_test, y_test)


def transform_image(file: Path) -> np.ndarray:
    """
    Loads the image into an array, then divides each square of the chess board
    into its own block, returning in the end an array of size (64, 50, 50, 3),
    where each of the 64 squares are represented in 50 pixels by 50 pixels
    across 3 color channels.

    Args:
        file (Path): path to the image file to be transformed

    Returns:
        np.ndarray: contains each of the 64 blocks in the chess board
    """
    image: np.ndarray = io.imread(file)
    size: int = int(image.shape[0] / 8)
    blocks: np.ndarray = view_as_blocks(image, (size, size, image.shape[2]))
   
    return blocks.squeeze(axis=2).reshape(64, block_size, block_size, 3)


def transform_label(label: str) -> np.ndarray:
    """
    Takes a FEN label and transforms it into a one-hot encoding.

    Args:
        label (str): FEN label for a given chess board

    Returns:
        np.ndarray: one-hot encoding of FEN label
    """
    symbols: str = "prbnkqPRBNKQ"

    eye: np.ndarray = np.eye(13)
    output: np.ndarray = np.empty((0, 13))

    label = re.sub("[-]", "", label)

    for char in label:
        if char in "12345678":
            output = np.append(output, np.tile(eye[12], (int(char), 1)), axis=0)
        else:
            index = symbols.index(char)
            output = np.append(output, eye[index].reshape((1, 13)), axis=0)

    return output
