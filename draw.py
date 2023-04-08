__author__ = 'Alaie Titor'

from typing import *
import argparse
import os
from typing import List
from PIL import Image, ImageDraw

__MAX_X__ = 5e4
__MAX_Y__ = 5e4

__COLORS__ = [
    (254, 75, 75),  # met1
    (81, 231, 75),  # met2
    (220, 247, 50),  # met3
    (187, 92, 124),  # met4
    (98, 227, 254),  # met5
    (0, 0, 153)  # met6
]


class Layer:
    def __init__(self, number: int) -> None:
        self.number = number
        self.resistors = []
        self.viases = []
        self.currents = []
        self.voltages = []


def get_rgb_from_range(value):
    range = 1
    rgb = [0, 0, 0]

    if(value >= 0 and value <= range/4):
        p = int(255*(value/(range/4)))

        rgb[0] = 0
        rgb[1] = p
        rgb[2] = 255

    if(value >= range/4 and value < range*2/4):
        p = int(255*((value - range/4)/(range/4)))

        rgb[0] = 0
        rgb[1] = 255
        rgb[2] = 255 - p

    if(value >= range*2/4 and value < range*3/4):
        p = int(255*((value - range*2/4)/(range/4)))

        rgb[0] = p
        rgb[1] = 255
        rgb[2] = 0

    if(value >= range*3/4 and value <= range):
        p = int(255*((value - range*3/4)/(range/4)))

        rgb[0] = 255
        rgb[1] = 255 - p
        rgb[2] = 0

    if(value > range):
        rgb[0] = 255

    if(rgb[0] == 0 and rgb[1] == 0 and rgb[2] == 0):
        print('Error', value, range)

    return (rgb[0], rgb[1], rgb[2])


def parse_spice(filepath: str):
    if os.path.exists(filepath):
        with open(filepath) as file:
            lines = file.readlines()
            layers: List[Layer] = [Layer(i + 1) for i in range(7)]

            min_max_x = [1e9, 0]
            min_max_y = [1e9, 0]

            for line in lines:
                line = line.lower()
                splited_line = line.split(' ')

                if (len(splited_line) == 4):
                    layer_number = int(splited_line[1][-1]) - 1
                    source = [int(x) for x in splited_line[1].split('_')[1:-1]] if splited_line[1] != '0' else [0, 0]
                    target = [int(x) for x in splited_line[2].split('_')[1:-1]] if splited_line[2] != '0' else [0, 0]
                    value = float(splited_line[-1])

                    # >--------- FIX
                    if(min_max_x[1] < source[0]):
                        min_max_x[1] = source[0]
                    if(min_max_x[0] > source[0] and splited_line[1] != '0'):
                        min_max_x[0] = source[0]

                    if(min_max_x[1] < target[0]):
                        min_max_x[1] = target[0]
                    if(min_max_x[0] > target[0] and splited_line[2] != '0'):
                        min_max_x[0] = target[0]

                    if(min_max_y[1] < source[1]):
                        min_max_y[1] = source[1]
                    if(min_max_y[0] > source[1] and splited_line[1] != '0'):
                        min_max_y[0] = source[1]

                    if(min_max_y[1] < target[1]):
                        min_max_y[1] = target[1]
                    if(min_max_y[0] > target[1] and splited_line[2] != '0'):
                        min_max_y[0] = target[1]
                    # FIX ---------<

                    if(splited_line[1][-1] == splited_line[2][-1] or splited_line[2] == '0' or splited_line[1] == '0'):
                        if(line[0] == 'r'):
                            layers[layer_number].resistors.append(source + target + [value])
                        if(line[0] == 'i'):
                            layers[layer_number].currents.append(source + target + [value])
                        if(line[0] == 'v'):
                            layers[layer_number].voltages.append(source + target + [value])
                    else:
                        layers[layer_number].viases.append(source + [value])

            return layers, min_max_x, min_max_y


def optimize_spice_image(layers: List[Layer], min_max_x, min_max_y):
    if(min_max_x[1] > __MAX_X__ or min_max_y[1] > __MAX_Y__):
        scale_factor_x = __MAX_X__/min_max_x[1] if min_max_x[1] > __MAX_X__ else 1
        scale_factor_y = __MAX_Y__/min_max_y[1] if min_max_y[1] > __MAX_Y__ else 1

        min_max_x = [x*scale_factor_x for x in min_max_x]
        min_max_y = [y*scale_factor_y for y in min_max_y]

        for layer in layers:
            for idx, param in enumerate(layer.resistors):
                layer.resistors[idx] = [param[0]*scale_factor_x, param[1]*scale_factor_y, param[2]*scale_factor_x, param[3]*scale_factor_y, param[-1]]

            for idx, param in enumerate(layer.viases):
                layer.viases[idx] = [param[0]*scale_factor_x, param[1]*scale_factor_y, param[-1]]

            for idx, param in enumerate(layer.currents):
                layer.currents[idx] = [param[0]*scale_factor_x, param[1]*scale_factor_y, param[-1]]

            for idx, param in enumerate(layer.voltages):
                layer.voltages[idx] = [param[0]*scale_factor_x, param[1]*scale_factor_y, param[2]*scale_factor_x, param[3]*scale_factor_y, param[-1]]

        return layers, min_max_x, min_max_y

    return layers, min_max_x, min_max_y


def draw_spice_topolgy(layers: List[Layer], min_max_x, min_max_y, outpath: str):
    padding = 500
    image = Image.new("RGB", size=(int(min_max_x[1] - min_max_x[0]) + padding*2, int(min_max_y[1] - min_max_y[0]) + padding*2), color=(0, 0, 0))
    draw = ImageDraw.Draw(image)

    for idx in range(len(layers)):
        layer = layers[idx]

        for current in layer.currents:
            _current = [current[0] - min_max_x[0] + padding, current[1] - min_max_y[0] + padding]

            draw.ellipse((_current[0] - 100, _current[1] - 100, (_current[0] + 100,  _current[1] + 100)), fill=(255, 0, 0))

    for idx in range(len(layers)):
        layer = layers[idx]

        for resistor in layer.resistors:
            _resistor = [resistor[0] - min_max_x[0] + padding, resistor[1] - min_max_y[0] + padding, resistor[2] - min_max_x[0] + padding, resistor[3] - min_max_y[0] + padding]

            draw.line(_resistor, fill=__COLORS__[layer.number - 1], width=50)
            draw.ellipse((_resistor[0] - 60, _resistor[1] - 60, (_resistor[0] + 60,  _resistor[1] + 60)), fill=(255, 255, 255))
            draw.ellipse((_resistor[2] - 60, _resistor[3] - 60, (_resistor[2] + 60,  _resistor[3] + 60)), fill=(255, 255, 255))

        for viases in layer.viases:
            _viases = [viases[0] - min_max_x[0] + padding, viases[1] - min_max_y[0] + padding]

            draw.ellipse((_viases[0] - 75, _viases[1] - 75, (_viases[0] + 75,  _viases[1] + 75)), fill=__COLORS__[layer.number - 1])

        for voltage in layer.voltages:
            _voltage = [voltage[0] - min_max_x[0] + padding, voltage[1] - min_max_y[0] + padding]

            draw.ellipse((_voltage[0] - 100, _voltage[1] - 100, (_voltage[0] + 100,  _voltage[1] + 100)), fill=(0, 255, 0))

    image.save(outpath, "PNG")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Draw image of topology from spice netlist.')
    parser.add_argument('-p', '--path', default=None, type=Union[None, str], help="Path to netlist of circuit.")
    parser.add_argument('-o', '--out', default=None, type=Union[None, str], help="Output path for image of topology.")
    namespace = parser.parse_args()

    path: str = namespace.path
    out: str = namespace.out

    if path == None:
        raise ValueError("Path is None!")

    if out == None:
        out = "/".join([x for x in (path[:-2] + ".png").split("/")])

    layers, min_max_x, min_max_y = parse_spice(path)
    layers, min_max_x, min_max_y = optimize_spice_image(layers, min_max_x, min_max_y)

    draw_spice_topolgy(layers, min_max_x, min_max_y, out)
