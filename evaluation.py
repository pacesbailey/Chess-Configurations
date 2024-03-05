import matplotlib.pyplot as plt

from keras.callbacks import History


def plot_training(history: History) -> None:
    """
    Plots the accuracy and loss for both the training and validation datasets 
    during the training process.

    Args:
        history (History): keras History object that stores the accuracy and 
                           loss values from the training process
    """
    figure, (axis_1, axis_2) = plt.subplots(2)

    axis_1.plot(history.history["accuracy"])
    axis_1.plot(history.history["loss"])
    axis_1.set_title("Training Accuracy")
    axis_1.set_xlabel("Epochs")
    axis_1.set_ylabel("Accuracy")
    axis_1.legend(["Accuracy", "Loss"])

    axis_2.plot(history.history["val_accuracy"])
    axis_2.plot(history.history["val_loss"])
    axis_2.set_title("Validation Accuracy")
    axis_2.set_xlabel("Epochs")
    axis_2.set_ylabel("Accuracy")
    axis_2.legend(["Validation Accuracy", "Validation Loss"])

    plt.tight_layout()
    plt.show()
