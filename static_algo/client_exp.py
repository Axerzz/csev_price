#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project:   csev
@File:      client_exp.py
@Author:    bai
@Email:     wenchao.bai@qq.com
@Date:      2021/10/24 23:45
@Purpose:   client of experiment module
"""

import numpy as np
import smooth
import multiprocessing as mp


def smooth_algo(region, cs, prov_flag=True):
    return smooth.smooth_algo_noV(region, cs, prov_flag)


region_num = 6
cs_num = 5

if __name__ == "__main__":
    """
    load data from npz file:

    npzfile = np.load('experiment/dat/regionFix_smooth_46_2_6_NJ.npz', allow_pickle=True)
    price_list_of_region4 = npzfile['prices'].item().get(4)
    """
    # run the scripts the get the experiment data

    # ans_dict = bcd(region_num, cs_num, True)
    # print("optimal price:", ans_dict["price"], file=w)
    # print("optimal revenue:", ans_dict["profit"], file=w)
    # print("time:", ans_dict["time"], file=w)

    ans_price = smooth_algo(region_num, cs_num, True)
    print("init_price:", ans_price)