import argparse
import keras_tuner as kt
import os
import tensorflow as tf

from keras.callbacks import EarlyStopping, History, ModelCheckpoint, \
    ReduceLROnPlateau
from pathlib import Path
from prettytable import PrettyTable
from evaluation import compare_models, plot_training
from exploration import display_image_statistics, display_label_statistics, \
    display_sample_image, get_image_statistics
from preprocessing import split_dataset
from model import generator, initialize_lenet5, initialize_simple, \
    initialize_vgg


DATA_PATH: Path = Path("dataset")
TRAIN_PATH: Path = DATA_PATH / "train"
TEST_PATH: Path = DATA_PATH / "test"
MODEL_PATH: Path = Path("models")

TRAIN_SIZE: int = 80_000
TEST_SIZE: int = 20_000
VAL_SPLIT: float = (1 / 8)
HT_SPLIT: float = (1 / 7)

IMAGE_SIZE: int = 224
BLOCK_SIZE: int = int(IMAGE_SIZE / 8)
INPUT_SHAPE: tuple = (BLOCK_SIZE, BLOCK_SIZE, 1)

SEED: int = 2024
EPOCHS: int = 30
tf.random.set_seed(SEED)


def main() -> None:
    """
    Given a dataset of chessboard images and their FEN notation labels, 
    amounting to a unique label for each image, this program trains,
    evaluates, and compares two models.
    """
    paths: tuple = split_dataset(TRAIN_PATH, VAL_SPLIT, HT_SPLIT)

    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        prog="Chess Configurations"
    )
    parser.add_argument(
        "--explore", "-x",
        dest="explore",
        help="Explores data set with visualizations",
        action="store_true",
        default=False
    )
    parser.add_argument(
        "--hypertune", "-ht", 
        dest="hypertune",
        help="Sets the program to hypertune a specified model",
        action="store_true",
        default=True
    )
    parser.add_argument(
        "--train", "-t", 
        dest="train",
        help="Sets the program to train a specified model",
        action="store_true",
        default=False
    )
    parser.add_argument(
        "--model", "-m", 
        dest="model",
        help="Chooses the model to be used",
        action="store",
        choices=["simple", "lenet5", "vgg"],
        default="simple"
    )
    parser.add_argument(
        "--evaluate", "-e", 
        dest="evaluate",
        help="Evaluates the trained model on the test data",
        action="store_true",
        default=False
    )
    args: Namespace = parser.parse_args()

    if args.explore:
        train_list: list = [TRAIN_PATH / file for file in os.listdir(TRAIN_PATH)]
        dimensions, filenames = get_image_statistics(train_list)
        display_image_statistics(dimensions)
        display_label_statistics(train_list, filenames)
        display_sample_image(paths[0])

    if args.hypertune:
        match args.model:
            case "simple":
                hypermodel = initialize_simple

            case "lenet5":
                hypermodel = initialize_lenet5

            case "vgg":
                hypermodel = initialize_vgg

        tuner: kt.BayesianOptimization = kt.BayesianOptimization(
            hypermodel=hypermodel,
            objective="accuracy",
            directory="models/hypertuning",
            project_name=f"{args.model}"
        )
        tuner.search(
            generator(paths[2], IMAGE_SIZE), 
            epochs=EPOCHS, 
            steps_per_epoch=len(paths[2]) // EPOCHS
        )

    if args.train:
        callbacks: list = [
            EarlyStopping(
                monitor="loss", mode="min",
                min_delta=0, patience=10,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=f"models/{args.model}.h5", 
                monitor="val_loss", mode="min",
                verbose=1, save_best_only=True
            )
        ]

        model: models.Sequential = tuner.hypermodel.build(
            tuner.get_best_hyperparameters(num_trials=1)[0]
        )
        history: History = model.fit(
            generator(paths[0], IMAGE_SIZE), 
            epochs=EPOCHS, 
            steps_per_epoch=len(paths[0]) // EPOCHS, 
            validation_data=generator(paths[1], IMAGE_SIZE), 
            validation_steps=len(paths[1]) // EPOCHS, 
            callbacks=callbacks
        )
        plot_training(history)

    if args.evaluate:
        scores: dict = compare_models(MODEL_PATH, TEST_PATH, IMAGE_SIZE)
        comparison_table: PrettyTable = PrettyTable()
        comparison_table.add_column("Models", list(scores.keys()))
        comparison_table.add_column(
            "Accuracy", 
            [score[1] for score in list(scores.values())]
        )
        comparison_table.add_column(
            "Loss", 
            [score[0] for score in list(scores.values())]
        )
        print(comparison_table)


if __name__ == "__main__":
    main()
