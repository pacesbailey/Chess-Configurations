from pathlib import Path
from dataloader import DataLoader


DATA_PATH: Path = Path("dataset")
TRAIN_PATH: Path = Path("dataset/train")
dataloader: DataLoader = DataLoader(DATA_PATH)
image_labels: list = dataloader.get_image_labels(TRAIN_PATH)
print(len(image_labels))
