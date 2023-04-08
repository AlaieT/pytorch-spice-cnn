__all__ = ["CircuitDataset"]

from typing import *
import os
import pandas as pd
from .reader import read
import torch
from torch.utils.data import Dataset
from pickle import dump, load
from tqdm import tqdm
import pickle
from torchvision import transforms as T
import random


class CircuitScaler():
    def __init__(self) -> None:
        self.min: List[float] = [0, 0, 0]
        self.max: List[float] = [0, 0, 0.0012]

    def set(self, matrix: torch.Tensor):
        z = matrix.shape[0]

        for i in range(z):
            _min = torch.min(matrix[i, :, :]).item()
            _max = torch.max(matrix[i, :, :]).item()

            if len(self.min) - 1 < i:
                self.min.append(_min)
            elif self.min[i] > _min:
                self.min[i] = _min

            if len(self.max) - 1 < i:
                self.max.append(_max)
            elif self.max[i] < _max:
                self.max[i] = _max

    def scale(self, matrix: torch.Tensor) -> torch.Tensor:
        z = matrix.shape[0]

        for i in range(z):
            data_std_r = (matrix[i, :, :] - self.min[i])/(self.max[i] - self.min[i])
            data_scaled_r = data_std_r * (1 - 0) + 0
            matrix[i, :, :] = data_scaled_r

        return matrix


class CircuitDataset(Dataset):
    def __init__(self, path_to_data: str, scaler_path: Optional[str] = None, train: bool = True, resave: bool = False) -> None:
        super().__init__()

        self.sources = []
        self.targets = []
        self.target_min = []
        self.scaler: CircuitScaler = CircuitScaler()

        if os.path.exists(path_to_data):
            df = pd.read_csv(path_to_data).to_numpy()

            if not(resave and train):
                with open(scaler_path, "rb") as file:
                    self.scaler = load(file)

            for i in tqdm(range(df.shape[0]), bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'):
                path = "/".join(df[i][0].split("/")[:-1])

                if not os.path.exists(path + "/tensors"):
                    os.mkdir(path + "/tensors")

                if resave:
                    matrix, target = read(df[i][0], df[i][1], size=(128, 352))

                    if train:
                        self.scaler.set(matrix)
                else:
                    matrix = torch.load(path + "/tensors/matrix.pt")
                    target = torch.load(path + "/tensors/target.pt")

                if(target != None):
                    self.sources.append(matrix)
                    self.targets.append(target)

                    min_target = target.flatten()
                    min_index = min_target.nonzero()
                    
                    self.target_min.append(torch.tensor([min_target[min_index].min().item()]))

            if (resave):
                for i in range(len(self.sources)):
                    path = "/".join(df[i][0].split("/")[:-1])
                    self.sources[i] = self.scaler.scale(self.sources[i])

                    torch.save(self.sources[i], path + "/tensors/matrix.pt")
                    torch.save(self.targets[i], path + "/tensors/target.pt")

            if (resave and train):
                with open(scaler_path, "wb") as file:
                    dump(self.scaler, file, pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
        if len(self.targets) > 0:
            return self.sources[idx], self.targets[idx], self.target_min[idx]
        else:
            return self.sources[idx], None
