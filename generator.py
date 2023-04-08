__author__ = 'Alaie Titor'

from typing import *
import argparse
from random import uniform, randint
from typing import List
import os
import pandas as pd
from copy import deepcopy


__METALL_RESITANCE__ = [
    26.041664e-5,  # l1
    8.1167556e-5,  # l2
    14.242278e-5,  # l3
    2.9375e-5,  # l4
    1.7777e-5,  # l5
]

__VIA_RESITANCE__ = [
    0.900000,  # 1-2
    0.852500,  # 3-4
    0.852500,  # 3-4
    0.380000  # 4-5
]


class BreakPoint:
    def __init__(self, type: str, x: int, y: int = None) -> None:
        self.type = type
        self.x = x
        self.y = y


class Line:
    def __init__(self, x_left: int, y_top: int, x_right: int, y_bottom: int) -> None:
        self.x_left = x_left
        self.x_right = x_right
        self.y_top = y_top
        self.y_bottom = y_bottom


class Resistor(Line):
    def __init__(self, value: float, x_left: int, y_top: int, x_right: int, y_bottom: int) -> None:
        super().__init__(x_left, y_top, x_right, y_bottom)

        self.value = value


class CurrentSource(Line):
    def __init__(self, value: float, x_left: int, y_top: int) -> None:
        super().__init__(x_left, y_top, 0, 0)

        self.value = value


class Layer:
    def __init__(self, mettal_value: float, number: int) -> None:
        self.number = number
        self.metall_value = mettal_value
        self.lines: List[Line] = []
        self.resistors: List[Resistor] = []

    def add_line(self, x_left: int, y_top: int, x_right: int, y_bottom: int):
        self.lines.append(Line(x_left, y_top, x_right, y_bottom))

    def add_resistor(self, x_left: int, y_top: int, x_right: int, y_bottom: int):
        value = (abs(x_left - x_right) if (x_right != x_left) else abs(y_top - y_bottom))*self.metall_value

        self.resistors.append(Resistor(value, x_left, y_top, x_right, y_bottom))

    def writeToFile(self, file):
        for idx, resistor in enumerate(self.resistors):
            file.write(F"R{self.number}{idx} VPWR_{resistor.x_left}_{resistor.y_top}_{self.number} VPWR_{resistor.x_right}_{resistor.y_bottom}_{self.number} {resistor.value}\n")


def remove_dublicated(break_points: List[BreakPoint], point="x"):
    fixed = []

    for i, b1 in enumerate(break_points):
        if(b1.type != "via"):
            if(i > 0 and i < len(break_points) - 1):
                if(abs(getattr(b1, point) - getattr(break_points[i+1], point)) > 0 and abs(getattr(b1, point) - getattr(break_points[i-1], point)) > 0):
                    fixed.append(deepcopy(b1))
            elif (i == 0):
                if(abs(getattr(b1, point) - getattr(break_points[i+1], point)) > 0):
                    fixed.append(deepcopy(b1))
            elif (i == len(break_points) - 1):
                if(abs(getattr(b1, point) - getattr(break_points[i-1], point)) > 0):
                    fixed.append(deepcopy(b1))
        else:
            fixed.append(deepcopy(b1))

    return fixed


def fix_space(break_points: List[BreakPoint], split_step):
    fixed = []

    for i, b1 in enumerate(break_points):
        if(b1.type != "via"):
            if(i > 0 and i < len(break_points) - 1):
                if(abs(b1.x - break_points[i+1].x) >= split_step*0.50 and abs(b1.x - break_points[i-1].x) >= split_step*0.50):
                    fixed.append(deepcopy(b1))
            elif (i == 0):
                if(abs(b1.x - break_points[i+1].x) >= split_step*0.50):
                    fixed.append(deepcopy(b1))
            elif (i == len(break_points) - 1):
                if(abs(b1.x - break_points[i-1].x) >= split_step*0.50):
                    fixed.append(deepcopy(b1))
        else:
            fixed.append(deepcopy(b1))

    return fixed


def writeVia(file, viases, _from, _to):
    for idx, via in enumerate(viases):
        res_value = __VIA_RESITANCE__[_from - 1]

        file.write(F"Rv{_from}{_to}{idx} VPWR_{int(via[0])}_{int(via[1])}_{_to+1} VPWR_{int(via[2])}_{int(via[3])}_{_from+1} {res_value}\n")


def generate_circuit(width, height, denisty_1, denisty_45, split_level, v_pos, current_values, pad, path):
    voltage = 1.8
    cell_height = int(height/(denisty_1))
    padding_x = int(width*pad)
    padding_y = int(width*pad)
    layers = [Layer(__METALL_RESITANCE__[i], i + 2) for i in range(5)]

    via_12 = []
    via_23 = []
    via_34 = []
    via_45 = []

    current_sources = []
    voltage_source = None

    # Get break points for 1'st layer:
    split_step = int((width - padding_x*0.1) / split_level)

    # Create layer 1:
    for i in range(denisty_1):
        y = i*(cell_height) + int(padding_y/3)

        layers[0].add_line(int(padding_x*0.05), y, width - int(padding_x*0.05), y)

    # Create layer 4:
    for i in range(denisty_45):
        x = int((width - padding_x*3) / (denisty_45 - 1)) * i + int(padding_x*0.05) + 2*split_step

        layers[3].add_line(x, 0, x, denisty_1*(cell_height) + int(padding_y/3))

    # Create layer 5:
    for i in range(denisty_1):
        if(i % int((denisty_1 + denisty_1/2)/denisty_45) == 0):
            y = i*(cell_height) + int(padding_y/3) + randint(int(cell_height*0.25), int(cell_height*0.75))

            layers[4].add_line(0, y, width, y)

    break_points_layer_1 = [BreakPoint("point", x*split_step) for x in range(split_level)] + [BreakPoint("via", line.x_left) for line in layers[3].lines]
    break_points_layer_1.sort(key=lambda point: point.x)
    break_points_layer_1 = fix_space(break_points_layer_1, split_step)

    for line in layers[0].lines:
        new_line = True

        for break_point in break_points_layer_1:
            _x = None
            _y = None

            if(break_point.x > line.x_left):
                if(new_line):
                    new_line = False
                    layers[0].add_resistor(line.x_left, line.y_top, break_point.x, line.y_bottom)
                else:
                    layers[0].add_resistor(layers[0].resistors[-1].x_right, layers[0].resistors[-1].y_bottom, break_point.x, line.y_bottom)

                _x = layers[0].resistors[-1].x_right
                _y = layers[0].resistors[-1].y_bottom

                if break_point.type == "via":
                    layers[1].add_resistor(_x, _y - 19, _x, _y)
                    layers[1].add_resistor(_x, _y, _x, _y + 19)

                    layers[2].add_resistor(_x - 79, _y, _x, _y)
                    layers[2].add_resistor(_x, _y, _x + 79, _y)

                    via_12.append([_x, _y, _x, _y])
                    via_23.append([_x, _y, _x, _y])
                    via_34.append([_x, _y, _x, _y])

            elif (break_point.x == line.x_left):
                _x = line.x_left
                _y = line.y_top

            # Create current source
            if(_x != None and _y != None):
                if(uniform(0, 1) > 0.1) or break_point.type == 'via':
                    if uniform(0, 1) > 0.05:
                        current_value = uniform(current_values[0], current_values[1])
                        current_sources.append(F"Ic{len(current_sources) + 1} VPWR_{_x}_{_y}_2 0 {current_value}\n")
                    else:
                        current_value = uniform(4e-12, 7e-12) if uniform(0, 1) > 0.35 else uniform(4e-11, 7e-11)
                        current_sources.append(F"If{len(current_sources) + 1} VPWR_{_x}_{_y}_2 0 {current_value}\n")

                    if uniform(0, 1) > 0.4:
                        current_sources.append(F"Ii{len(current_sources) + 1} VPWR_{_x}_{_y}_2 0 {current_value}\n")

        layers[0].add_resistor(layers[0].resistors[-1].x_right, layers[0].resistors[-1].y_bottom, line.x_right, line.y_bottom)

    # Get break points for 4'st layer:
    break_points_layer_4 = [BreakPoint("point", line.x_left, line.y_top) for line in layers[0].lines] + [BreakPoint("via", line.x_left, line.y_top) for line in layers[4].lines]
    break_points_layer_4.sort(key=lambda point: point.y)
    break_points_layer_4 = remove_dublicated(break_points_layer_4, "y")

    for line in layers[3].lines:
        new_line = True

        for break_point in break_points_layer_4:
            if(line.x_left >= break_point.x):
                if(new_line):
                    new_line = False
                    layers[3].add_resistor(line.x_left, line.y_top, line.x_right, break_point.y)
                else:
                    layers[3].add_resistor(layers[3].resistors[-1].x_right, layers[3].resistors[-1].y_bottom, line.x_right, break_point.y)

                if break_point.type == "via":
                    via_45.append([layers[3].resistors[-1].x_right, layers[3].resistors[-1].y_bottom, layers[3].resistors[-1].x_right, layers[3].resistors[-1].y_bottom])

        layers[3].add_resistor(layers[3].resistors[-1].x_right, layers[3].resistors[-1].y_bottom, line.x_right, line.y_bottom)

    if voltage_source == None:
        voltage_source = [via_45[v_pos][0], via_45[v_pos][1], voltage]

    # Get break points for 5'th layer:
    break_points_layer_5 = [BreakPoint("point", line.x_left) for line in layers[3].lines]
    break_points_layer_5.sort(key=lambda point: point.x)
    break_points_layer_5 = remove_dublicated(break_points_layer_5)

    for line in layers[4].lines:
        new_line = True

        for break_point in break_points_layer_5:
            if(break_point.x >= line.x_left):
                if(new_line):
                    new_line = False
                    layers[4].add_resistor(line.x_left, line.y_top, break_point.x, line.y_bottom)
                else:
                    layers[4].add_resistor(layers[4].resistors[-1].x_right, layers[4].resistors[-1].y_bottom, break_point.x, line.y_bottom)

        layers[4].add_resistor(layers[4].resistors[-1].x_right, layers[4].resistors[-1].y_bottom, line.x_right, line.y_bottom)

    with open(path, "w") as file:
        # VOLTAGE
        file.write(F"V0 VPWR_{voltage_source[0]}_{voltage_source[1]}_6 0 {voltage_source[2]}\n")

        # VIA
        writeVia(file, via_12, 1, 2)
        writeVia(file, via_23, 2, 3)
        writeVia(file, via_34, 3, 4)
        writeVia(file, via_45, 4, 5)

        # LAYERS
        for layer in layers:
            layer.writeToFile(file)

        # CURRENT
        for source in current_sources:
            file.write(source)

        file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate circuits.')
    parser.add_argument("-p", "--path", default="./assets/train", help="Path to dataset.")
    parser.add_argument("-w", "--width", nargs='+', default=[300000, 350000, 400000], help='Range of width in micrometres.')
    parser.add_argument("-d1", "--denisty_1", nargs='+', default=[30, 35, 40, 45, 55], help='Denisty of std cell lines in 1st layer of metall.')
    parser.add_argument("-d45", "--denisty_45", nargs='+', default=[2, 3, 4, 6, 8], help='Denisty of metall lines in 4 and 5th layers.')
    parser.add_argument("-s", "--split_level", nargs='+', default=[15, 20, 25, 30], help='Ammount of span in std cell lines in 1st layers of metall.')
    parser.add_argument("-p", "--padding", nargs='+', default=[15, 20, 25, 30], help='Padding of x and y axis of 4-5 layers in procentage of width.')
    namespace = parser.parse_args()

    path = namespace.path
    _width = namespace.width
    _denisty_1 = namespace.denisty_1
    _denisty_45 = namespace.denisty_45
    _split_level = namespace.split_level
    _padding = namespace.padding

    files: List[str] = []
    count = 0

    for width in _width:
        for denisty_1 in _denisty_1:
            for denisty_45 in _denisty_45:
                for split_level in _split_level:
                    for current_values in [[0.075e-5, 0.125e-4], [0.15e-5, 0.25e-4], [0.3e-5, 0.5e-4], [0.6e-5, 1.00e-4], [1.2e-5, 2.00e-4]]:
                        for pading in _padding:
                            for v_pos in range(denisty_45*2):
                                count += 1

                                if not os.path.exists(F"{path}/pdsim_{count}"):
                                    os.mkdir(F"{path}/pdsim_{count}")

                                generate_circuit(width, width, denisty_1, denisty_45,  split_level, v_pos, current_values,
                                                 pading,  F"{path}/pdsim_{count}/pdsim_{count}.sp")

                                files.append([os.path.join(F"{path}", f'pdsim_{count}/pdsim_{count}.sp'),
                                              os.path.join(F"{path}", f'pdsim_{count}/pdsim_{count}.csv')])

                                if os.path.exists(os.path.join(F"{path}", f'pdsim_{count}/pdsim_{count}.csv')):
                                    os.remove(os.path.join(F"{path}", f'pdsim_{count}/pdsim_{count}.csv'))

    df = pd.DataFrame(data=files, columns=['Source', 'Target'])
    df.to_csv(os.path.join("/".join([x for x in path.split("/")[:-1]]), f'{path.split("/")[-1]}.csv'), index=False)
