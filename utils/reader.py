__all__ = ["read"]

from typing import *
import os
import pandas as pd
import numpy as np
import torch
from copy import deepcopy

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class Element():
    def __init__(self, x_start: int, y_start: int, x_end: int, y_end: int, type: str, value: Optional[Union[float, None]] = None) -> None:
        self.x_start = x_start
        self.x_end = x_end
        self.y_start = y_start
        self.y_end = y_end
        self.type = type
        self.value = 0 if None else value


class Layer():
    def __init__(self, number: int) -> None:
        self.number = number
        self.elements: Dict[str, List[Element]] = dict()

    def add_element(self, key: str, element: Element) -> None:
        if self.elements.get(key) == None:
            self.elements[key] = [element]
        else:
            self.elements[key].append(element)

    def sort_elements(self, key_to_sort: str = "x_start") -> None:
        sorted_elements: Dict[str, List[Element]] = dict()

        keys = list(self.elements)
        keys.sort(key=lambda x: int(x))

        for key in keys:
            sorted_elements[key] = deepcopy(self.elements[key])
            sorted_elements[key].sort(key=lambda x: getattr(x, key_to_sort))

        self.elements = sorted_elements


def sort_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    sorted = dict()
    keys = list(d)

    keys.sort(key=lambda x: int(x))

    for key in keys:
        sorted[key] = deepcopy(d[key])

    return sorted


def set_dict_indexes(d: Dict[str, Any]) -> int:
    for idx, key in enumerate(d):
        d[key] = idx*2

    return idx*2


def read_spice(file_path: str) -> Union[List[Layer], None]:
    if os.path.exists(file_path):
        with open(file_path, 'r') as spice_file:
            lines = spice_file.readlines()

            x_points = dict()
            y_points = dict()
            layers: List[Layer] = [Layer(i) for i in range(7)]

            for line in lines:
                line = line.lower()

                if(line[0] != '*' and line[0] != '.'):
                    splited_line = line.split(' ')

                    value = float(splited_line[-1])
                    start_pin = [int(x) for x in splited_line[1].split("_")[1:]]
                    end_pin = [int(x) for x in splited_line[2].split("_")[1:]] if splited_line[2] != "0" else [0, 0, 0]

                    if splited_line[2] != "0":
                        if start_pin[0] > end_pin[0] or start_pin[1] > end_pin[1]:
                            tmp = end_pin
                            end_pin = start_pin
                            start_pin = tmp

                        x_points[str(end_pin[0])] = None
                        y_points[str(end_pin[1])] = None

                    x_points[str(start_pin[0])] = None
                    y_points[str(start_pin[1])] = None

                    layer_number = (start_pin[-1] if start_pin[-1] > end_pin[-1] else end_pin[-1]) - 2
                    layer_key = str(start_pin[0]) if start_pin[0] == end_pin[0] and (layer_number == 1 or layer_number == 3) else str(start_pin[1])

                    if line[0] == 'r':
                        type = "element" if start_pin[-1] == end_pin[-1] else "via"
                        layers[layer_number].add_element(layer_key, Element(start_pin[0], start_pin[1], end_pin[0], end_pin[1], type, value))
                    elif line[0] == 'i':
                        layers[5].add_element(layer_key, Element(start_pin[0], start_pin[1], start_pin[0], start_pin[1], "via", value))
                    elif line[0] == 'v':
                        layers[6].add_element(layer_key, Element(start_pin[0], start_pin[1], start_pin[0], start_pin[1], "via", value))

            for layer, key in zip(layers, ["x_start", "y_start", "x_start", "y_start", "x_start", "x_start", "x_start"]):
                layer.sort_elements(key)

            x_points = sort_dict(x_points)
            y_points = sort_dict(y_points)

            return layers, x_points, y_points
    else:
        raise ValueError(f"Path: {file_path} does not exists!")


def read_csv(file_path: str):
    if os.path.exists(file_path):
        df = pd.read_csv(file_path).to_numpy()
        data: Dict[str, float] = dict()

        for i in range(df.shape[0]):
            splited_line = [int(x) for x in df[i][0].split("_")[1:]]

            if len(splited_line) != 0:
                data[F"{splited_line[0]}_{splited_line[1]}_{int(df[i][0][-1])-2}"] = (df[i][1] - 1.6) / (1.8 - 1.6)

        return data

    else:
        raise ValueError(f"Path: {file_path} does not exists!")


def create_matrix(layers: List[Layer], data: Union[Dict[str, float], None], x_points: Dict[str, None], y_points: Dict[str, None], max_x: int, max_y: int) -> Tuple[np.ndarray, Union[np.ndarray, None]]:

    matrix = np.zeros((3, max_x, max_y))
    target = np.zeros((1, max_x, max_y)) if data != None else None

    for i, z in zip([0, 1, 2, 3, 4, 5, 6], [1, 1, 1, 1, 1, 2, 0]):
        for key_line in layers[i].elements:
            if i in [1, 3]:
                x = x_points[key_line]

                for element in layers[i].elements[key_line]:
                    y_start = y_points[str(element.y_start)]
                    y_end = y_points[str(element.y_end)]

                    if element.type == "via":
                        matrix[z, x, y_start] += np.array([element.value])
                        
                        # New
                        if data != None  and i < 5:
                            target[0, x, y_start] = data[F"{key_line}_{element.y_start}_{i}"]
                    else:
                        matrix[z, x, y_start + 1:y_end] += np.array([element.value/(y_end - y_start - 1)]*(y_end - y_start - 1))
                        
                        # New
                        if data != None  and i < 5:
                            start_v = data[F"{key_line}_{element.y_start}_{i}"]
                            end_v = data[F"{key_line}_{element.y_end}_{i}"]
                            step_v = abs(start_v - end_v)/(y_end - y_start)

                            if start_v <= end_v and (y_end - y_start) > 1:
                                target[0, x, y_start + 1:y_end] = np.array([start_v + x*step_v for x in range(1, y_end - y_start)])
                            else:
                                target[0, x, y_start + 1:y_end] = np.array([start_v - x*step_v for x in range(1, y_end - y_start)])

                            target[0, x, y_start] = data[F"{key_line}_{element.y_start}_{i}"]
                            target[0, x, y_end] = data[F"{key_line}_{element.y_end}_{i}"]
            else:
                y = y_points[key_line]

                for element in layers[i].elements[key_line]:
                    x_start = x_points[str(element.x_start)]
                    x_end = x_points[str(element.x_end)]

                    if element.type == "via":
                        matrix[z, x_start, y] += np.array([element.value])

                        # New
                        if data != None and i < 5:
                            target[0, x_start, y] = data[F"{element.x_start}_{key_line}_{i}"]
                    else:
                        matrix[z, x_start + 1:x_end, y] += np.array([element.value/(x_end - x_start - 1)]*(x_end - x_start - 1))

                        # New
                        if data != None and i < 5:
                            start_v = data[F"{element.x_start}_{key_line}_{i}"]
                            end_v = data[F"{element.x_end}_{key_line}_{i}"]
                            step_v = abs(start_v - end_v)/(x_end - x_start)

                            if start_v <= end_v and (x_end - x_start) > 1:
                                target[0, x_start + 1:x_end, y] = np.array([start_v + x*step_v for x in range(1, x_end - x_start)])
                            else:
                                target[0, x_start + 1:x_end, y] = np.array([start_v - x*step_v for x in range(1, x_end - x_start)])

                            target[0, x_start, y] = data[F"{element.x_start}_{key_line}_{i}"]
                            target[0, x_end, y] = data[F"{element.x_end}_{key_line}_{i}"]

    return matrix, target



def read(source_path: str, target_path: Optional[str] = None, size: Tuple[int, int] = (384, 384)) -> Tuple[Union[torch.Tensor, None], Union[torch.Tensor, None]]:
    x_points = None
    y_points = None
    max_x = None
    max_y = None

    data = None

    matrix = None
    target = None


    if target_path != None and os.path.exists(target_path):
        data = read_csv(target_path)

    if os.path.exists(source_path):
        layers, x_points, y_points = read_spice(source_path)
        max_x = set_dict_indexes(x_points)
        max_y = set_dict_indexes(y_points)

        if max_x <= size[0]:
            max_x = size[0]
        else:
            raise ValueError(F"Target size is smaller then actual!, X:{max_x}")

        if max_y <= size[1]:
            max_y = size[1]
        else:
            raise ValueError(F"Target size is smaller then actual!, Y:{max_y}")

        matrix, target = create_matrix(layers, data, x_points, y_points, max_x, max_y)

        matrix = torch.from_numpy(matrix).float()
        target = torch.from_numpy(target).float() if data != None else None

    return matrix, target
