from pathlib import Path


class DataLoader():
    def __init__(self, dataset_path: Path):
        self.dataset_path = dataset_path
    
    def get_image_labels(self, directory_path: Path) -> list[str]:
        """
        Gets the file names of each image in the given directory and returns 
        them in a list to be then used as categories.

        Args:
            directory_path (Path): directory where image files are located

        Returns:
            list[str]: contains the FEN labels for each image
        """
        image_labels: list = []
        for file in directory_path.iterdir():
            if file.is_file():
                image_labels.append(file.stem)
        
        return image_labels
    
    def load_images(self):
        pass

    
