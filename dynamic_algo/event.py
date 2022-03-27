#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/8/1
# @Author : zyx
# @Version：V 0.1
# @File : config.py
# @desc : configuration file
import static_algo.config_exp

config = static_algo.config_exp.Config()
import numpy as np


# 时间类，一共有3中类型的事件，区域车辆生成，车辆到达，车辆开始充电，车辆充电完毕离开
class event:
    charging = [0 for i in range(config.cs_num)]
    nums_cars_cs = [0 for i in range(config.cs_num)]

    def __init__(self, time, id, type="generate", lam=100, nums=0):
        self.time = time
        self.id = id
        self.type = type
        self.nums = nums
        if type == "generate":
            self.num_cars_region = [np.random.poisson(lam) for i in range(config.region_num)]
