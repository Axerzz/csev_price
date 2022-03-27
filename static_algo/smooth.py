#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project:   csev
@File:      smooth.py
@Author:    bai
@Email:     wenchao.bai@qq.com
@Date:      2021/11/15 14:36
@Purpose:   init strategy matrix using smooth algo.
"""

from static_algo.config_exp import config
from pyomo.environ import *
from pyomo.mpec import *


def smooth_algo_V(region_num, cs_num, ev_num, prov_flag=True):
    vehicle_vector = ev_num
    return init_price(region_num, cs_num, vehicle_vector, prov_flag)


def smooth_algo_noV(region_num, cs_num, prov_flag=True):
    vehicle_vector = config.vehicle_vector
    return init_price(region_num, cs_num, vehicle_vector, prov_flag)


def init_price(region_num, cs_num, vehicle_vector, prov_flag=True):
    config.change_region_cs_num(region_num=region_num, cs_num=cs_num)

    N = config.region_num
    M = config.cs_num
    cs_cap = config.cs_cap_vector

    if prov_flag:
        dist_vector = config.dist_vector
    else:
        dist_vector = config.wuhan_dist_vector

    model = ConcreteModel()
    model.priceIndex = range(M)  # 表示研究的是所有的充电站的定价策略
    model.region = range(N)
    model.CS = range(M)

    model.p = Var(model.priceIndex, bounds=(config.P_MIN, config.P_MAX))
    model.f = Var(model.CS, model.region, bounds=(0.0, 1.0))
    model.z = Var(model.CS, model.region, bounds=(0.0, 1.0))
    model.v = Var(model.CS, model.region, bounds=(0, None))
    model.lamuda = Var(model.region)

    model.obj = Objective(
        expr=sum((model.p[k] - config.cost[k]) * sum(vehicle_vector[i] * model.f[k, i] for i in model.region)
                 for k in model.priceIndex), sense=maximize)

    # 原问题的等式线性约束，一共有n个
    model.single_f = ConstraintList()
    for i in model.region:
        model.single_f.add(sum(model.f[j, i] for j in model.CS) == 1.0)

    # 拉格朗日乘子约束，一共有m * n个
    model.lagrange = ConstraintList()
    for i in model.region:
        for j in model.CS:
            model.lagrange.add(config.w_p * model.p[j] + config.w_d * dist_vector[j][i] + config.w_qf * (
                        model.f[j, i] * vehicle_vector[i] +
                        sum(model.f[j, k] * vehicle_vector[k] for k in model.region)) / cs_cap[j] -
                               model.v[j, i] - model.lamuda[i] == 0.0)

    # 互补约束，一共有m * n个
    model.compl = ComplementarityList()
    for i in model.region:
        for j in model.CS:
            model.compl.add(complements(model.f[j, i] >= 0, model.v[j, i] >= 0))

    # smooth constraint
    model.smooth = ConstraintList()
    for i in model.region:
        for j in model.CS:
            model.smooth.add(model.f[j, i] - model.z[j, i] == 0)
            model.smooth.add(sqrt((model.z[j, i] - model.v[j, i]) ** 2 + 1e-10) - model.z[j, i] - model.v[j, i] == 0)

    opt = SolverFactory('ipopt')
    opt.solve(model)

    p_list = []

    for j in model.priceIndex:
        p_list.append(round(model.p[j](), 3))

    return p_list


if __name__ == "__main__":
    print(init_price(6, 6, True))

