"""
Created: 11/6/19
© Denisa Qori McDonald 2019 All Rights Reserved
"""
import math
from math import floor
from typing import List

import numpy as np

from BioHCI.knitted_components.knitted_component import KnittedComponent


class UniformTouchpad(KnittedComponent):
    def __init__(self, num_rows: int, num_cols: int, total_resistance: float, button_resistance:
    float, inter_button_resistance: float, inter_col_resistance: float):
        self.__name = "UniformTouchpad"
        self.__num_rows = num_rows
        self.__num_cols = num_cols

        self.__total_resistance = total_resistance
        self.__button_resistance = button_resistance
        self.__inter_button_resistance = inter_button_resistance
        self.__inter_col_resistance = inter_col_resistance

        calculated_total_resistance = round(
            self.num_buttons * self.button_resistance + self.num_inter_button * self.inter_button_resistance +
            self.num_inter_col * self.inter_col_resistance)

        assert (total_resistance * 0.8 <= calculated_total_resistance) or (total_resistance * 1.2 >=
                                                                           calculated_total_resistance), \
            "The measured total resistance seems to be higher than the calculated one, even with a 20% error threshold."

        self.__button_centers = self.__produce_button_centers()

    @property
    def name(self):
        return self.__name

    @property
    def num_rows(self) -> int:
        return self.__num_rows

    @property
    def num_cols(self) -> int:
        return self.__num_cols

    @property
    def num_buttons(self) -> int:
        return self.num_rows * self.num_cols

    @property
    def num_inter_button(self) -> int:
        return self.num_rows * (self.num_cols - 1)

    @property
    def num_inter_col(self) -> int:
        return self.num_rows - 1

    @property
    def total_resistance(self) -> float:
        return self.__total_resistance

    @property
    def button_resistance(self) -> float:
        return self.__button_resistance

    @property
    def inter_button_resistance(self) -> float:
        return self.__inter_button_resistance

    @property
    def inter_col_resistance(self) -> float:
        return self.__inter_col_resistance

    @property
    def button_centers(self) -> List[float]:
        return self.__button_centers

    # -- SETTERS -- #
    @total_resistance.setter
    def total_resistance(self, resistance: float):
        self.__total_resistance = resistance

    @num_rows.setter
    def num_rows(self, num: int):
        self.__num_rows = num

    @num_cols.setter
    def num_cols(self, num: int):
        self.__num_cols = num

    @button_resistance.setter
    def button_resistance(self, resistance: float):
        self.__button_resistance = resistance

    @inter_button_resistance.setter
    def inter_button_resistance(self, resistance: float):
        self.__inter_button_resistance = resistance

    @inter_col_resistance.setter
    def inter_col_resistance(self, resistance: float):
        self.__inter_col_resistance = resistance

    def get_button_id(self, est_location: float) -> int:
        min_idx = 0
        min_dist = math.fabs(self.button_centers[0] - est_location)
        for i, center in enumerate(self.button_centers):
            dist = math.fabs(center - est_location)
            if dist < min_dist:
                min_dist = dist
                min_idx = i

        return min_idx

    def __get_button_center(self, id: int) -> float:
        return self.button_centers[id]

    def get_button_centers(self, ids: np.ndarray):
        centers = []
        for id in ids:
            centers.append(self.__get_button_center(id))

        centers = np.asarray(centers)
        return centers

    def __produce_button_centers(self) -> List[float]:
        centers_raw = []
        for button in range(0, self.num_buttons):
            center_resistance = self.button_resistance * (button + 0.5) + self.inter_button_resistance * (
                    (self.num_cols - 1) * floor(button / self.num_cols) + (
                    button % self.num_cols)) + self.inter_col_resistance * floor(button / self.num_cols)

            centers_raw.append(center_resistance)

        centers_norm = self.__scale(centers_raw)
        return centers_norm

    def get_row_labels(self, button_list: np.ndarray) -> np.ndarray:
        row_cat_list = []
        for button in button_list:
            row_cat = math.floor(button/3)
            row_cat_list.append(row_cat)
        return np.asarray(row_cat_list)

    @staticmethod
    def __scale(raw_values: List[float]) -> List[float]:
        min = raw_values[0]
        max = raw_values[0]

        for val in raw_values:
            if val < min:
                min = val
            elif val > max:
                max = val

        norm_ls = []
        for val in raw_values:
            norm_val = (val - min) / (max - min)
            norm_ls.append(norm_val)

        return norm_ls


if __name__ == "__main__":
    touchpad = UniformTouchpad(num_rows=12, num_cols=3, total_resistance=534675,
                               button_resistance=7810.0, inter_button_resistance=4590.0, inter_col_resistance=13033.0)
    print(touchpad.button_centers)
    id1 = touchpad.get_button_id(4300000.4)
    id2 = touchpad.get_button_id(14098.3)
    id3 = touchpad.get_button_id(43000.4)
    id4 = touchpad.get_button_id(62000.4)
    id5 = touchpad.get_button_id(67543.4)
    id6 = touchpad.get_button_id(68907.4)

    print(id1, id2, id3, id4, id5, id6)
