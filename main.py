import argparse
import tensorflow as tf

from keras.callbacks import EarlyStopping, History, ModelCheckpoint, \
    ReduceLROnPlateau
from pathlib import Path
from evaluation import plot_training
from exploration import display_sample_image
from preprocessing import split_validation_data
from model import generator, initialize_simple, initialize_vgg


DATA_PATH: Path = Path("dataset")
TRAIN_PATH: Path = DATA_PATH / "train"
TEST_PATH: Path = DATA_PATH / "test"

TRAIN_SIZE: int = 80_000
TEST_SIZE: int = 20_000
SPLIT: float = 0.2

IMAGE_SIZE: int = 224
BLOCK_SIZE: int = int(IMAGE_SIZE / 8)
INPUT_SHAPE: tuple = (BLOCK_SIZE, BLOCK_SIZE, 1)

SEED: int = 2024
EPOCHS: int = 100
tf.random.set_seed(SEED)


def main() -> None:
    """
    Given a dataset of chessboard images and their FEN notation labels, 
    amounting to a unique label for each image, this program trains,
    evaluates, and compares two models.
    """
    train_filepaths, val_filepaths = split_validation_data(TRAIN_PATH, SPLIT)

    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        prog="Chess Configurations"
    )
    parser.add_argument(
        "--explore", "-x",
        dest="explore",
        help="Explores data set with visualizations",
        action="store_true",
        required=False,
        default=False
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
        choices=["simple", "vgg16"],
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

    if args.explore:
        display_sample_image(TRAIN_PATH)

    if args.train:
        callbacks: list = [
            EarlyStopping(
                monitor="val_loss", mode="min",
                min_delta=0, patience=10
            ),
            ReduceLROnPlateau(
                monitor="val_loss", mode="min",
                min_lr=0.001, patience=5
            ),
            ModelCheckpoint(
                filepath=f"models/{args.model}.h5", 
                monitor="val_loss", mode="min",
                verbose=1, save_best_only=True
            )
        ]
        
        match args.model:
            case "simple":
                model: layers.Sequential = initialize_simple(INPUT_SHAPE)

            case "vgg16":
                model: layers.Sequential = initialize_vgg(INPUT_SHAPE)

        model.summary()
        history: History = model.fit(
            generator(train_filepaths, IMAGE_SIZE),
            epochs=EPOCHS,
            steps_per_epoch=len(train_filepaths) / EPOCHS,
            validation_data=generator(val_filepaths, IMAGE_SIZE),
            validation_steps=len(val_filepaths) / EPOCHS,
            callbacks=callbacks
        )

        if args.evaluate:
            plot_training(history)


if __name__ == "__main__":
    main()
