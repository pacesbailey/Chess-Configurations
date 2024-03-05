import numpy as np
import re

from pathlib import Path
from skimage import color, io, transform
from skimage.util.shape import view_as_blocks


def preprocess_image(file: Path, size: int) -> np.ndarray:
    """
    Loads the image into an array, downsamples it to the specified size,
    converts it from an RGB image into a grayscale one, then divides the
    chessboard into blocks, each representing one of the 8 x 8 squares, 
    finally being returned as an array of the dimensions (64 x 28 x 28 x 1).

    Args:
        file (Path): path to the image file to be transformed
        size (int): size to reduce the image to

    Returns:
        np.ndarray: contains each of the 64 blocks in the chess board
    """
    image: np.ndarray = io.imread(file)
    image = transform_image(image, size)

    # Separates chessboard squares into individual arrays
    block_size: int = int(size / 8)
    blocks: np.ndarray = view_as_blocks(image, (block_size, block_size))
   
    return blocks.reshape(64, block_size, block_size, 1)


def encode_label(label: str) -> np.ndarray:
    """
    Takes a FEN label and transforms it into a one-hot encoding.

    Args:
        label (str): FEN label for a given chess board

    Returns:
        np.ndarray: one-hot encoding of FEN label
    """
    label = re.sub("[-]", "", label)

    # Symbols which represent chess pieces in the FEN labels
    symbols: str = "prbnkqPRBNKQ"

    eye: np.ndarray = np.eye(13)
    output: np.ndarray = np.empty((0, 13))
    
    for char in label:
        if char in "12345678":
            output = np.append(output, np.tile(eye[12], (int(char), 1)), axis=0)
        else:
            index = symbols.index(char)
            output = np.append(output, eye[index].reshape((1, 13)), axis=0)

    return output


def transform_image(image: np.ndarray, size: int) -> np.ndarray:
    """
    Given an image in an array and a specified downsample size, reduces the
    images dimensions, normalizes it, then converts the image from RGB to 
    grayscale.

    Args:
        image (np.ndarray): image to be downsampled
        size (int): dimension to downsample the image to

    Returns:
        np.ndarray: downsampled, normalized, and grayscale image
    """
    resized = transform.resize(image, (size, size), mode="constant")
    gray = color.rgb2gray(resized)

    return (gray - np.min(gray)) / (np.max(gray) - np.min(gray))


def main() -> None:
    pass


if __name__ == "__main__":
    main()
