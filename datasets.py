from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
from torchvision import datasets
from PIL import Image as PILImage
import os, torch, pandas as pd


# Functions for the datasets
class Datasets:
    @staticmethod
    def mnist() -> DataLoader:
        mnist: datasets.MNIST = datasets.MNIST(
            root="datasets", train=True, download=True, transform=ToTensor()
        )
        return DataLoader(mnist, batch_size=64, shuffle=True)

    @staticmethod
    def fromcsv(csv: str, path: str) -> DataLoader:
        data: CSVDataset = CSVDataset(csv_file=csv, root_dir=path, transform=ToTensor())
        return DataLoader(data, batch_size=64, shuffle=True)

    @staticmethod
    def fromtuple(data: list[tuple[torch.Tensor, torch.Tensor]]) -> DataLoader:
        dataset: TupleDataset = TupleDataset(data)
        return DataLoader(dataset, batch_size=64, shuffle=True)


# Custom csv dataset
class CSVDataset(Dataset):
    def __init__(self, csv_file: str, root_dir: str, transform=None) -> None:
        self.transform = transform
        self.root_dir: str = root_dir
        self.csv_path: str = os.path.join(self.root_dir, csv_file)
        self.csv_file: pd.DataFrame = pd.read_csv(self.csv_path)

    def __len__(self) -> int:
        return len(self.csv_file)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path: str = f"{self.root_dir}/images/{self.csv_file.iloc[index, 0]}"
        image = PILImage.open(img_path).convert("RGB")
        label: torch.Tensor = torch.tensor(self.csv_file.iloc[index, 1])

        if self.transform:
            sample: torch.Tensor = self.transform(image)
            return (sample, label)
        return (torch.empty(0), torch.empty(0))


# Custom tuple dataset
class TupleDataset(Dataset):
    def __init__(self, data: list[tuple[torch.Tensor, torch.Tensor]]) -> None:
        self.data: list[tuple[torch.Tensor, torch.Tensor]] = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx]
