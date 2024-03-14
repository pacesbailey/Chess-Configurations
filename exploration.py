import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import random

from pathlib import Path
from PIL import Image
from preprocessing import transform_image


def get_image_statistics(directory: list) -> tuple[list, list]:
    """
    Collects the dimensions and FEN labels for each of the images in the
    training data.

    Args:
        directory (list): contains Path objects that lead to the images

    Returns:
        tuple[list, list]: contains the dimensions and filenames
    """
    dimensions: list = []
    filenames: list = []
    for file in directory:
        try:
            with Image.open(file) as image:
                width, height = image.size
                dimensions.append((width, height))
                filenames.append(file.stem)
        except Exception as e:
            print(f"Error processing {file}: {e}")

    return dimensions, filenames


def display_image_statistics(dimensions: list) -> None:
    """
    Displays the collected image statistics in two subplots as histograms.

    Args:
        dimensions (list): contains height-width pairs in tuples
    """
    widths: list = [dimension[0] for dimension in dimensions]
    heights: list = [dimension[1] for dimension in dimensions]
    
    plt.figure(figsize=(15, 5))
    plt.suptitle("Dimensions")

    plt.subplot(1, 2, 1)
    plt.hist(widths, bins=20)
    plt.title("Widths")
    plt.xlabel("Pixels")
    plt.ylabel("Images")

    plt.subplot(1, 2, 2)
    plt.hist(heights, bins=20)
    plt.title("Heights")
    plt.xlabel("Pixels")
    plt.ylabel("Images")

    plt.show()


def display_label_statistics(directory: list, filenames: list) -> None:
    n_images: int = len(directory)
    n_unique_labels: int = len(set(filenames))
    print(f"For {n_images} images, there are {n_unique_labels} unique FEN labels.")


def display_sample_image(directory: list) -> None:
    """
    Given a directory filled with images of chessboards, displays a sample
    image along with the associated FEN label.

    Args:
        directory (list): contains paths for training images
    """
    image_name: Path = random.choice(directory)
    image: np.ndarray = mpimg.imread(image_name)

    transformed: np.ndarray = transform_image(image, 224)

    fig, (axis_1, axis_2) = plt.subplots(1, 2)
    fig.suptitle(f"Sample Image: {image_name.stem}")
    axis_1.imshow(image)
    axis_1.set_title("Original")
    axis_1.axis("off")
    axis_2.imshow(transformed, cmap="gray")
    axis_2.set_title("Transformed")
    axis_2.axis("off")
    plt.show()


def main() -> None:
    pass


if __name__ == "__main__":
    main()
