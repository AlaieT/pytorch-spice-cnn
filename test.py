__author__ = 'Alaie Titor'

from typing import *
import argparse
from utils import read
from pickle import load
import torch
import segmentation_models_pytorch as smp
import seaborn as sns
import matplotlib.pyplot as plt
import os
import time


def mape(y_pred: torch.Tensor, y_true: torch.Tensor):
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()

    index = y_true.nonzero()

    loss = (y_pred[index] - y_true[index]) / y_true[index]

    return loss.abs().mean() * 100


def get_min(target):
    tmp = target.flatten()

    return torch.min((tmp[tmp.nonzero()]*(1.8 - 1.6) + 1.6)/1.8).item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test nn with ground truth.')
    parser.add_argument("-p", "--path", default="./assets/test", help="Path to dataset.")
    parser.add_argument("-s", "--scaler", default="./dict/scaler/scaler.pkl", help="Path to scaler.")
    parser.add_argument("-m", "--model", default='./dict/dnn/best.pt', help="Path to model.")
    namespace = parser.parse_args()

    path = namespace.path
    path_scaler = namespace.scaler
    path_model = namespace.model

    tests = os.listdir(path)

    with open(path_scaler, "rb") as file:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        scaler = load(file)
        aux_params = dict(pooling='avg', dropout=0, activation=None, classes=1)
        model = smp.Unet(encoder_name="resnet50", encoder_weights="imagenet", in_channels=3, classes=1, activation=None, aux_params=aux_params).eval().to(device)
        model.load_state_dict(torch.load(path_model))

        preds = []

        for test in tests:
            start_time = time.time()
            name = test.split("/")[-1].split(".")[0]

            print(F"Testing - {name}")

            matrix, target = read(F"{path}/{test}/{name}.sp", F'{path}/{test}/{name}.csv', size=(128, 352))
            matrix, target = scaler.scale(matrix.to(device)).unsqueeze(0), target.to(device)

            with torch.no_grad():
                prediction, min_prediction = model(matrix)

                target = target.squeeze().squeeze()
                prediction = prediction.squeeze().squeeze()

                # Gettin values from resistor mask
                nonzero_indexes = (target == 0).nonzero()

                prediction[nonzero_indexes[:, 0], nonzero_indexes[:, 1]] = 0
                target[nonzero_indexes[:, 0], nonzero_indexes[:, 1]] = 0

                # Removin padding
                prediction = prediction
                prediction = prediction[~torch.all(prediction == 0, axis=1)]
                prediction = prediction[:, ~torch.all(prediction == 0, axis=0)]
                prediction = torch.where(prediction > 0, (prediction*(1.8 - 1.6) + 1.6)/1.8, 0)
                min_prediction = ((min_prediction*(1.8 - 1.6) + 1.6)/1.8).item()

                target = target
                target = target[~torch.all(target == 0, axis=1)]
                target = target[:, ~torch.all(target == 0, axis=0)]
                target = torch.where(target > 0, (target*(1.8 - 1.6) + 1.6)/1.8, 0)

                min_target = target.flatten()
                min_index = min_target.nonzero()
                min_target = min_target[min_index].min().item()

                # Calculate loss
                loss = mape(prediction, target).item()
                preds.append(loss)
                print(F"MAPE: {loss:.5}%, MIN_MAPE: {abs(min_prediction - min_target)/min_target*100:.5}%, TIME: {time.time() - start_time}\n")

                min_target = torch.min(target.flatten()[target.flatten().nonzero()]).item()

                prediction = prediction.cpu().numpy()
                target = target.cpu().numpy()

                heatmap = sns.heatmap(data=prediction, cmap="coolwarm_r", vmin=min_target, vmax=1, mask=(prediction == 0))
                heatmap.set_facecolor("black")

                plt.savefig(F'{path}/{test}/prediction.png')
                plt.close()

                heatmap = sns.heatmap(data=target, cmap="coolwarm_r", vmin=min_target, vmax=1, mask=(target == 0))
                heatmap.set_facecolor("black")

                plt.savefig(F'{path}/{test}/target.png')
                plt.close()

        print(f"MEAN MAPE: {sum(preds)/len(preds)}%")
