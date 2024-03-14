import matplotlib.pyplot as plt
import os

from keras.callbacks import History
from keras.models import load_model
from pathlib import Path
from typing import Generator
from model import test_generator
from preprocessing import process_dataset


def plot_training(history: History) -> None:
    """
    Plots the accuracy and loss for both the training and validation datasets 
    during the training process.

    Args:
        history (History): keras History object that stores the accuracy and 
                           loss values from the training process
    """
    figure, (axis_1, axis_2) = plt.subplots(2)

    # Training subplot
    axis_1.plot(history.history["accuracy"])
    axis_1.plot(history.history["loss"])
    axis_1.set_title("Training Accuracy")
    axis_1.set_xlabel("Epochs")
    axis_1.set_ylabel("Accuracy")
    axis_1.legend(["Accuracy", "Loss"])

    # Validation subplot
    axis_2.plot(history.history["val_accuracy"])
    axis_2.plot(history.history["val_loss"])
    axis_2.set_title("Validation Accuracy")
    axis_2.set_xlabel("Epochs")
    axis_2.set_ylabel("Accuracy")
    axis_2.legend(["Validation Accuracy", "Validation Loss"])

    plt.tight_layout()
    plt.show()


def compare_models(model_dir: Path, test_dir: Path, image_size: int) -> dict:
    """
    Evaluates the models' performance on the test dataset, then compares them
    to each other.

    Args:
        model_dir (Path): filepath to directory containing the trained models
        test_dir (Path): filepath to directory containing the test dataset
        image_size (int): size to which images must be reduced
    """
    test_files: list = [test_dir / file for file in os.listdir(test_dir)]
    model_names: list[str] = ["simple", "lenet5", "vgg"]
    scores: dict = {}
    
    for name in model_names:
        print("-" * 100)
        print(f"Evaluating {name.upper()} Model".center(100))
        print("-" * 100)
        model: models.Sequential = load_model(model_dir / f"{name}.h5")
        scores[name] = model.evaluate(test_generator(test_files, image_size))
        print("\n")

    return scores


def main() -> None:
    pass


if __name__ == "__main__":
    main()
