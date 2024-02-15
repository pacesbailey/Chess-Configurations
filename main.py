from pathlib import Path
from preprocessing import preprocess_dataset


TRAIN_PATH: Path = Path("data/train")
TEST_PATH: Path = Path("data/test")


def main():
    x_train, y_train, x_test, y_test = preprocess_dataset(TRAIN_PATH, TEST_PATH)


if __name__ == "__main__":
    main()
