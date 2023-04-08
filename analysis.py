__author__ = 'Alaie Titor'

from typing import *
import argparse
import pandas as pd
import numpy as np
import os


def generated_data_analysis(path: str, threshold: Union[None, str], out: Union[None, str]):
    df = pd.read_csv(path)
    new_df = [] if out != None and threshold != None else None

    targets = df.iloc[:, 1]
    sources = df.iloc[:, 0]

    print(f'\nData length: {df.shape[0]}')

    diff = np.array([])

    for source, target in zip(sources, targets):
        if os.path.exists(target):
            df_target = pd.read_csv(target)
            values = df_target.iloc[:, 1]

            if np.max(values) != 0:
                curretn_diff = 1 - values[values > 0]/np.max(values)
                diff = np.append(diff, curretn_diff)

                if(new_df != None and np.max(curretn_diff) < 0.11):
                    new_df.append([source, target])
                    
            else:
                print("Found negative numbers: ", target)

    print(f'Diff - mean: {np.mean(diff)*100:.8}% max: {np.max(diff)*100:.8}% min: {np.min(diff)*100:.8}%')

    if new_df:
        pd.DataFrame(data=new_df, columns=["Source", "Target"]).to_csv(out, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Anlaysis irdrop of dataset(min, mean and max values)')
    parser.add_argument('-p', '--path', default='./assets/train.csv', help='Path to .csv of dataset.')
    parser.add_argument('-t', '--threshold', default=None, type=float, help='The threshold hold value for dataset filter.')
    parser.add_argument('-o', '--out', default=None, help='Out path for filtered dataset')
    namespace = parser.parse_args()

    path = namespace.path
    threshold = float(namespace.threshold)
    out = namespace.out

    generated_data_analysis(path, threshold, out)
