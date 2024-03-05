import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from preprocessing import transform_image


def display_sample_image(directory: Path) -> None:
    """
    Given a directory filled with images of chessboards, displays a sample
    image along with the associated FEN label.

    Args:
        directory (Path): filepath leading to a directory with images
    """
    image_name: Path = next(directory.iterdir())
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
    train_directory: Path = Path("dataset/train")
    display_sample_image(train_directory)


if __name__ == "__main__":
    main()
