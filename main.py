import argparse

from keras.callbacks import History
from pathlib import Path
from evaluation import plot_training
from model import generator, initialize_simple


DATA_PATH: Path = Path("dataset")
TRAIN_PATH: Path = DATA_PATH / "train"
TEST_PATH: Path = DATA_PATH / "test"

IMAGE_SIZE: int = 224
BLOCK_SIZE: int = int(IMAGE_SIZE / 8)

TRAIN_SIZE: int = 80_000
TEST_SIZE: int = 20_000
PERCENTAGE: float = 1.0

EPOCHS: int = 100


def main() -> None:
    """
    Given a dataset of chessboard images and their FEN notation labels, 
    amounting to a unique label for each image, this program trains,
    evaluates, and compares two models.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        prog="Chess Configurations"
    )
    
    parser.add_argument(
        "--train", "-t", 
        dest="train",
        help="Sets the program to train a specified model",
        action="store_true",
        required=True,
        default=True
    )
    
    parser.add_argument(
        "--model", "-m", 
        dest="model",
        help="Chooses the model to be used",
        action="store",
        required=True,
        choices=["simple"],
        default="simple"
    )

    parser.add_argument(
        "--evaluate", "-e", 
        dest="evaluate",
        help="Evaluates the trained model on the test data",
        action="store_true",
        required=True,
        default=True
    )
    
    args: Namespace = parser.parse_args()

    if args.train:
        if args.model == "simple":
            model: layers.Sequential = initialize_simple((BLOCK_SIZE, BLOCK_SIZE, 1))
            model.summary()
            history: History = model.fit(
                generator(TRAIN_PATH, IMAGE_SIZE, PERCENTAGE),
                epochs=EPOCHS,
                steps_per_epoch=(TRAIN_SIZE * PERCENTAGE) / EPOCHS,
                validation_data=generator(TEST_PATH, IMAGE_SIZE, PERCENTAGE),
                validation_steps=(TEST_SIZE * PERCENTAGE) / EPOCHS
            )
            model.save("models/simple.keras")

        if args.evaluate:
            plot_training(history)


if __name__ == "__main__":
    main()
