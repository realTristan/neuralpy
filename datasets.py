from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
from torchvision import datasets
import pandas as pd
import os, torch, PIL

# Datasets
class Datasets:
    @staticmethod
    def mnist() -> DataLoader:
        mnist: datasets.MNIST = datasets.MNIST(
            root="datasets", train=True, download=True, transform=ToTensor()
        )
        return DataLoader(mnist, batch_size=64, shuffle=True)
    
    @staticmethod
    def fromcsv(csv: str, path: str)-> DataLoader:
        data: CSVDataset = CSVDataset(
            csv_file=csv, root_dir=path, transform=ToTensor()
        )
        return DataLoader(data, batch_size=64, shuffle=True)


# Custom csv dataset
class CSVDataset(Dataset):
    def __init__(self, csv_file: str, root_dir: str, transform=None) -> None:
        self.transform = transform
        self.root_dir: str = root_dir
        self.csv_path: str = os.path.join(self.root_dir, csv_file)
        self.csv_file: pd.DataFrame = pd.read_csv(self.csv_path)

    def __len__(self) -> None:
        return len(self.csv_file)

    def __getitem__(self, idx) -> None:
        img_path = os.path.join(self.root_dir, "images", self.csv_file.iloc[idx, 0])
        image: PIL.Image = PIL.Image.open(img_path).convert("RGB")
        label: torch.Tensor = torch.tensor(self.csv_file.iloc[idx, 1])

        if self.transform:
            sample = self.transform(image)
        return (sample, label)


class TupleDataset(Dataset):
    def __init__(self, data: list[tuple[torch.Tensor, torch.Tensor]]) -> None:
        self.data: list[tuple[torch.Tensor, torch.Tensor]] = data

    def __len__(self) -> None:
        return len(self.data)

    def __getitem__(self, idx) -> None:
        return self.data[idx]