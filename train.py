__author__ = 'Alaie Titor'

import argparse
import os
import random
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import gc
from typing import *
from utils import CircuitDataset, plot_loss
import segmentation_models_pytorch as smp
from math import inf


class L2Error(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        input, target = input.view(-1), target.view(-1)
        loss = (input - target).pow(2).sum().sqrt() / target.pow(2).sum().sqrt()

        return loss


def set_seed(
    seed: int = 777
):
    '''Set seed for every random generator that used in project'''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    os.environ['PYTHONHASHSEED'] = str(seed)


def mape(y_pred: torch.Tensor, y_true: torch.Tensor):
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()

    index = y_true.nonzero()

    loss = (y_pred[index] - y_true[index]) / y_true[index]

    return loss.abs().mean() * 100


def train_step(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    dataloader: DataLoader,
    device: torch.device
):
    model.train()
    train_metrics = torch.empty((0, 2))

    for source, target, min in tqdm(dataloader, bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}', total=len(dataloader)):
        optimizer.zero_grad()

        source, target, min = source.to(device), target.to(device), min.to(device)

        prediction, prediction_min = model(source)
        loss = criterion(prediction, target, prediction_min, min)

        loss.backward()

        optimizer.step()
        scheduler.step()

        metrics = torch.Tensor([[loss.item(), mape(prediction, target)]])
        train_metrics = torch.cat([train_metrics, metrics], dim=0)

    del source, target, prediction, loss, metrics
    gc.collect()

    return train_metrics.mean(dim=0).tolist()


def valid_step(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device
):
    model.eval()
    valid_metrics = torch.empty((0, 1))

    with torch.no_grad():
        for source, target, _ in tqdm(dataloader, bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}', total=len(dataloader)):

            source, target = source.to(device), target.to(device)
            prediction, _ = model(source)

            metrics = torch.Tensor([[mape(prediction, target)]])
            valid_metrics = torch.cat([valid_metrics, metrics], dim=0)

        del source, target, prediction, metrics
        gc.collect()

    return valid_metrics.mean(dim=0).tolist()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model training')
    parser.add_argument('-ft', '--file_train', default='./assets/train.csv', help='File path to train .csv file.')
    parser.add_argument('-fv', '--file_valid', nargs='+', default=[], help='File path to train .csv file.')
    parser.add_argument('-e', '--epochs', default=1500, type=int, help='Count of train epochs.')
    parser.add_argument('-bt', '--batch_size_train', default=128, type=int, help='Size of train batch.')
    parser.add_argument('-bv', '--batch_size_valid', default=128, type=int, help='Size of valid batch.')
    parser.add_argument('-r', '--resave', nargs='?', const=True, help='Resave train and batch data.')
    namespace = parser.parse_args()

    file_train = namespace.file_train
    file_valid = namespace.file_valid
    batch_size_train = namespace.batch_size_train
    batch_size_valid = namespace.batch_size_valid
    epochs = namespace.epochs
    resave = namespace.resave

    if(len(file_valid) != 0):
        for valid in file_valid:
            assert os.path.exists(valid), f'Path to {valid} does not existss!'

    assert os.path.exists(file_train), 'Train dataset is not exists'
    assert batch_size_train > 0, 'Batch size not allowed to be 0'
    assert batch_size_valid > 0, 'Batch size not allowed to be 0'
    assert epochs > 0, 'Epochs not allowed to be 0'

    set_seed(seed=42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    aux_params = dict(pooling='avg', dropout=0.0, activation=None, classes=1)
    model = smp.Unet(encoder_name="resnet50", encoder_weights="imagenet", in_channels=3, classes=1, activation=None, aux_params=aux_params).to(device)
    model.load_state_dict(torch.load("./dict/dnn/last.pt"))
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1-5e-5)

    l2 = L2Error()

    def criterion(prediction: torch.Tensor, target: torch.Tensor, prediction_min: torch.Tensor, target_min: torch.Tensor):
        l2_prediction = prediction.flatten()
        l2_target = target.flatten()
        l2_index = l2_target.nonzero()

        return l2(l2_prediction[l2_index], l2_target[l2_index]) + l2(prediction_min, target_min)

    metric = np.zeros((0, 2 + len(file_valid)))
    best_loss = +inf

    folds_dataloders: List[DataLoader] = []

    dataset_train = CircuitDataset(file_train, scaler_path='./dict/scaler/scaler.pkl', train=True, resave=resave)
    dataloader_train = DataLoader(dataset=dataset_train, batch_size=batch_size_train, shuffle=True)

    for valid in file_valid:
        dataset_valid_out = CircuitDataset(valid, scaler_path='./dict/scaler/scaler.pkl', train=False, resave=True)
        folds_dataloders.append(DataLoader(dataset=dataset_valid_out, batch_size=batch_size_valid, shuffle=False))

    print(f'\nStart traning on {epochs} epochs\n')

    for i in range(epochs):
        train_mean_acc = train_step(model, criterion, optimizer, scheduler, dataloader_train, device)

        folds_mean_acc = []
        for dataloader in folds_dataloders:
            folds_mean_acc += valid_step(model, dataloader, device)

        metric = np.append(metric, [train_mean_acc + folds_mean_acc], axis=0)

        torch.save(model.state_dict(), './dict/dnn/_last.pt')

        if(metric[-1, 2] < best_loss):
            best_loss = metric[-1, 2]
            torch.save(model.state_dict(), './dict/dnn/_best.pt')

        print(f'\n---> {i+1}\033[95m LR:\033[0m {optimizer.param_groups[0]["lr"]:3e}')
        print(f'| Train \033[94mL2Error:\033[0m {metric[-1, 0]:.5}')
        print(f'| Train \033[94mMAPE\033[0m: {metric[-1, 1]:.5}%')

        for idx in range(len(file_valid)):
            print(f'| Fold{idx} \033[96mMAPE\033[0m: {metric[-1, 2 + idx]:.5}%')

        print('\033[94m--------------------------------------------------------------------------------\033[0m')

        if (i + 1) % 2 == 0:
            plot_loss(dataset=metric[:, :1].T, titles=['train'], save_path='./plots/train.png')
            plot_loss(dataset=metric[:, [1, ] + [2 + idx for idx in range(len(file_valid))]].T,
                      titles=['train mape %'] + [f'fold{idx} mape %' for idx in range(len(file_train))], save_path='./plots/mape.png')
