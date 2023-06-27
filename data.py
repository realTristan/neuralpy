from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import pandas as pd
import os, torch
from PIL import Image


class Data:
    def __init__(self, csv: str = None, path: str = None):
        if csv is None and path is None:
            self.model = self.mnist()
        else:
            self.model = self.custom(csv, path)

    # MNIST dataset
    def mnist(self):
        mnist: datasets.MNIST = datasets.MNIST(
            root="data", train=True, download=True, transform=ToTensor()
        )
        return DataLoader(mnist, batch_size=64, shuffle=True)

    # Custom dataset
    def custom(self, csv: str, path: str):
        custom: CustomDataset = CustomDataset(
            csv_file=csv, root_dir=path, transform=ToTensor()
        )
        return DataLoader(custom, batch_size=64, shuffle=True)


# Custom dataset
class CustomDataset(Dataset):
    def __init__(self, csv_file: str, root_dir: str, transform = None) -> None:
        self.transform = transform
        self.root_dir: str = root_dir
        self.csv_path: str = os.path.join(self.root_dir, csv_file)
        self.csv_file: pd.DataFrame = pd.read_csv(self.csv_path)

    def __len__(self) -> None:
        return len(self.csv_file)

    def __getitem__(self, idx) -> None:
        img_path = os.path.join(self.root_dir, "images", self.csv_file.iloc[idx, 0])
        image: Image = Image.open(img_path).convert("RGB")
        label: torch.Tensor = torch.tensor(self.csv_file.iloc[idx, 1])

        if self.transform:
            sample = self.transform(image)
        return (sample, label)
