#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project:   csev
@File:      config_exp.py
@Author:    bai
@Email:     wenchao.bai@qq.com
@Date:      2021/10/25 10:55
@Purpose:   a more convenience config for experiment
"""

import numpy as np

class Config:
    """配置文件，包含相关环境数据等

    """

    def __init__(self):
        super().__init__()
        self.threshold = 0.000001  # 迭代截止阈值
        self.total_cs = 13
        self.total_regions = 13

        self.total_price = np.array([85, 90, 80, 60, 55, 65, 68, 74, 85, 48, 58, 40, 90])
        self.total_vehicle_vector = np.array([559, 167, 350, 346, 5, 200, 80, 60, 20, 92, 70, 130])
        # self.total_vehicle_vector = np.array([75, 153, 182, 33, 358, 208, 140, 80, 60, 20, 92, 70, 130])

        # 横坐标是CS维度，纵坐标是区域维度
        self.total_dist_vector = np.array([[14.6514, 43.4912, 85.8734, 66.0878, 50.4677, 36.9472],
                                           [71.0449, 98.1383, 47.689, 9.68888, 40.0168, 38.0073],
                                           [75.8642, 57.7501, 24.645, 37.8972, 10.7528, 42.8276],
                                           [114.517, 65.1346, 14.0155, 40.8803, 49.4096, 81.4797],
                                           [46.9808, 7.71284, 53.5447, 87.9221, 50.3968, 72.7247]])

        self.wuhan_total_dist_vector = np.array([[15, 20, 23, 35, 40, 45, 60, 66, 79, 84, 79, 93, 100],
                                                 [25, 10, 35, 42, 50, 55, 69, 66, 76, 88, 90, 97, 104],
                                                 [40, 35, 12, 25, 40, 48, 59, 68, 77, 82, 88, 98, 110],
                                                 [45, 40, 38, 15, 40, 55, 44, 68, 75, 85, 83, 91, 99],
                                                 [60, 48, 42, 55, 10, 15, 30, 46, 61, 75, 81, 94, 102],
                                                 [65, 60, 55, 47, 38, 14, 35, 49, 56, 68, 79, 88, 98],
                                                 [68, 55, 50, 37, 44, 30, 10, 38, 59, 62, 78, 85, 86],
                                                 [72, 63, 60, 58, 50, 38, 40, 14, 38, 59, 77, 90, 102],
                                                 [80, 73, 63, 57, 64, 46, 60, 57, 17, 50, 70, 85, 98],
                                                 [88, 75, 70, 66, 51, 40, 38, 28, 40, 11, 32, 99, 120],
                                                 [96, 90, 88, 80, 76, 70, 65, 58, 44, 30, 10, 49, 78],
                                                 [98, 80, 88, 70, 66, 50, 45, 38, 44, 20, 30, 13, 68],
                                                 [106, 90, 78, 87, 71, 60, 55, 48, 44, 40, 38, 28, 10]]).transpose((1, 0))

        self.cs_cap_vector = [10, 4, 4, 5, 6, 8, 12, 8, 6, 6, 6, 4, 12]
        self.w_p, self.w_qf, self.w_d = 0.7, 0.1, 0.2
        self.P_MIN, self.P_MAX = 40, 90

        self.region_num = 6  # 下层区域的数量6
        self.cs_num = 5  # 充电站的数量5
        self.cost = [65, 60, 25, 15, 20, 15, 24, 36, 27, 13, 22, 28, 25]  # 运营成本

        # 根据充电站数量和研究区域的数量截取相关环境数据
        self.price = self.total_price[:self.cs_num].tolist()
        self.cs_cap = self.cs_cap_vector[:self.cs_num]
        self.vehicle_vector = self.total_vehicle_vector[:self.region_num].tolist()
        self.dist_vector = self.total_dist_vector[:self.cs_num, :self.region_num].tolist()
        self.wuhan_dist_vector = self.wuhan_total_dist_vector[:self.cs_num, :self.region_num].tolist()

    def change_region_num(self, change_num):
        self.region_num = change_num

    def change_cs_num(self, change_num):
        self.cs_num = change_num

    def change_region_cs_num(self, region_num, cs_num):
        self.region_num = region_num
        self.cs_num = cs_num
        self.price = self.total_price[:self.cs_num].tolist()
        self.vehicle_vector = self.total_vehicle_vector[:self.region_num].tolist()
        self.cs_cap = self.cs_cap_vector[:self.cs_num]
        self.dist_vector = self.total_dist_vector[:self.cs_num, :self.region_num].tolist()
        self.wuhan_dist_vector = self.wuhan_total_dist_vector[:self.cs_num, :self.region_num].tolist()


config = Config()
