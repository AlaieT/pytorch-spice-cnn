__author__ = 'Alaie Titor'

from typing import *
import argparse
from utils import read
from pickle import load
import torch
import segmentation_models_pytorch as smp
import seaborn as sns
import matplotlib.pyplot as plt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test nn with ground truth.')
    parser.add_argument("-p", "--path", default="./assets/test", help="Path to netlist file")
    parser.add_argument("-s", "--scaler", default="./dict/scaler/scaler.pkl", help="Path to scaler.")
    parser.add_argument("-m", "--model", default='./dict/dnn/last.pt', help="Path to model.")
    parser.add_argument("-t", "--threshold", default=0.11, help="The threshold hold value of trained model.")
    namespace = parser.parse_args()

    path = namespace.path
    path_scaler = namespace.scaler
    path_model = namespace.model
    threshold = namespace.threshold

    with open(path_scaler, "rb") as file:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        scaler = load(file)
        aux_params = dict(pooling='avg', dropout=0, activation=None, classes=1)
        model = smp.Unet(encoder_name="resnet50", encoder_weights="imagenet", in_channels=3, classes=1, activation=None, aux_params=aux_params).eval().to(device)
        model.load_state_dict(torch.load(path_model))

        matrix, _ = read(source_path=path, size=(128, 384))
        matrix = scaler.scale(matrix.to(device)).unsqueeze(0)

        with torch.no_grad():
            prediction, min_prediction = model(matrix)

            prediction = prediction.squeeze().squeeze()

            # Gettin values from resistor mask
            nonzero_indexes = (matrix[0, 1, :, :] == 0).nonzero()

            prediction[nonzero_indexes[:, 0], nonzero_indexes[:, 1]] = 0

            # Removin padding
            prediction = prediction
            prediction = prediction[~torch.all(prediction == 0, axis=1)]
            prediction = prediction[:, ~torch.all(prediction == 0, axis=0)]
            prediction = torch.where(prediction > 0, (prediction*(1.8 - 1.8*(1 - threshold)) + 1.8*(1 - threshold))/1.8, 0)
            min_prediction = (min_prediction*(1.8 - 1.8*(1 - threshold)) + 1.8*(1 - threshold)).item()

            prediction = prediction.cpu().numpy()

            print(F"Min voltage value: {min_prediction}")

            heatmap = sns.heatmap(data=prediction, cmap="coolwarm_r", vmin=0, vmax=1, mask=(prediction == 0))
            heatmap.set_facecolor("black")

            plt.savefig(F'{"/".join(path.split("/")[:-1])}/prediction.png')
            plt.close()
