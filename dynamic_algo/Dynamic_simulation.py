#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2022/3/24
# @Author : lyw
# @Version：V 0.1
# @File : Dynamic_simulation.py
# @desc : dynamic_simulation

import numpy as np
import random
from event import *
from cal_best_cs import *
from Queue1 import *

from static_algo.EV_Equilibrium_only_equal import EvEquilibrium
from static_algo.config_exp import config
from static_algo import smooth

evEquilibrium = EvEquilibrium()
import math

charging_required_time = 20
time_slices = 520
charging_rate = 0.5
per_step_time = 5
time_interval = 250
part_road = [[] for i in range(6)]


def get_part_road():
    for i in range(1, 7):
        s = '../partition_road/partition' + str(i) + '.txt'
        f = open(s)
        line = f.readline()
        while line:
            line = line.strip("\n")
            part_road[i - 1].append(line)
            line = f.readline()
        f.close()


def simulations(price_interval):
    f = open("../flow.txt")
    w = open("../new_flow.txt", "w+")
    print("[", file=w)
    line = f.readline()
    price = smooth.smooth_algo_noV(config.region_num, config.cs_num, True)
    # 赶路事件
    cs_queue_arrive = My_PriorityQueue()
    # 充电事件
    cs_queue_over = My_PriorityQueue()
    # 排队车辆数
    cs_inqueue = [0 for i in range(config.cs_num)]
    # 剩余充电容量
    cs_capacity = config.cs_cap
    car_speed = 15
    revenue = [0 for i in range(config.cs_num)]
    revenue_cs = [[0 for i in range(config.cs_num)] for j in range(int(time_slices / per_step_time))]
    price_time = []
    price_time.append(price)

    # 当前时间片之前所有时间内各区域的总车辆数
    vehicle_num_last_interval = [0 for i in range(config.region_num)]
    t = 0
    revenue_predict_time = []

    line = f.readline()
    line = line.strip("\n")
    while t < time_slices:  # 时间片分割，每1 step一个时间片
        print("现在是第", t, "个时间片")
        # 预处理
        while not cs_queue_arrive.empty():
            event_now = cs_queue_arrive.pop()
            if event_now.time <= t:
                id_now = event_now.id - 1
                cs_inqueue[id_now] += event_now.nums
                continue
            cs_queue_arrive.push(event_now, event_now.time)
            print("到达事件处理完成")
            break
        while not cs_queue_over.empty():
            event_now = cs_queue_over.pop()
            if event_now.time <= t:
                id_now = event_now.id - 1
                cs_capacity[id_now] += event_now.nums
                continue
            cs_queue_over.push(event_now, event_now.time)
            print("充电结束事件处理完成")
            break
        for i in range(config.cs_num):
            if cs_capacity[i] >= cs_inqueue[i] != 0:
                cs_queue_over.push(
                    event(t + charging_required_time, i + 1, type='over', nums=cs_inqueue[i]),
                    t + charging_required_time)  # 生成对应的充电完成事件并插入队列
                cs_capacity[i] -= cs_inqueue[i]
                cs_inqueue[i] = 0
            else:
                if cs_capacity[i] < cs_inqueue[i]:
                    cs_queue_over.push(
                        event(t + charging_required_time, i + 1, type='over', nums=cs_capacity[i]),
                        t + charging_required_time)  # 生成对应的充电完成事件并插入队列
                    cs_inqueue[i] -= cs_capacity[i]
                    cs_capacity[i] = 0
        print("各CS剩余容量：", cs_capacity)

        # 当前时间片下的各区域新出现的车辆数
        vehicle_num_now = [0 for i in range(config.region_num)]
        # 车流分配
        while line != "]":
            flow_time = get_flow_time(line)
            if flow_time <= t:
                rand = random.random()
                if rand > charging_rate:
                    print(line, file=w)
                    line = f.readline()
                    line = line.strip("\n")
                    continue
                start_pos = get_start_pos(line)
                end_pos = get_end_pos(line)
                # 计算到各个充电站的成本函数 假设best_cs是最优选择
                # (经纬度距离计算 路径更改 判断所属区域）
                # ————————————————————————————————————————————
                # 判断所属区域
                region_in = 0
                for i in range(6):
                    if start_pos in part_road[i]:
                        region_in = i
                        break
                # 寻找best_cs
                best_cs, dist, intersection, dist1, dist2 = get_best_cs(start_pos, end_pos, price, cs_inqueue)
                print("start_pos : ", start_pos)
                print("end_pos : ", end_pos)
                print("region_in : ", region_in)
                print("start_intersection : ", intersection)
                print("best_cs : ", best_cs)
                print("total_dist : ", dist)
                print("dist1 : ", dist1)
                print("dist2 : ", dist2)
                cs_queue_arrive.push(event(math.ceil(t + dist / car_speed), best_cs, type="arrive", nums=1),
                                     math.ceil(t + dist / car_speed))
                revenue[best_cs] += price[best_cs] - config.cost[best_cs]
                revenue_cs[int(t / per_step_time)][best_cs] += price[best_cs] - config.cost[best_cs]
                vehicle_num_now[region_in] += 1
                vehicle_num_last_interval[region_in] += 1
                # 路径更改，输出到新的json文件中
                new_route = change_route(line, best_cs, start_pos, end_pos)
                print("new_route: " + new_route)
                print(new_route, file=w)
                line = f.readline()
                line = line.strip("\n")
            else:
                break

        # 生成strategy
        print("各电站的总收益为:", revenue)
        t = t + per_step_time
        if not (t % price_interval):  # 每price_interval个时间片更新一次价格，超出最大时间片数则为静态价格
            print(vehicle_num_last_interval)
            # 使用smooth算法更新价格，主要影响因素是上阶段各区域的车辆数
            price = smooth.smooth_algo_V(config.region_num, config.cs_num, vehicle_num_last_interval)
            vehicle_num_last_interval = [0 for i in range(config.region_num)]
            price_time.append(price)
    print("]", file=w)
    print(revenue_cs)
    print(revenue)
    print(price_time)
    return revenue, revenue_cs, revenue_predict_time


def change_route(line, best_cs, start_road, end_road):
    cs_road = get_best_cs_road(start_road, best_cs)
    new_line = line.split("[")[0] + '[' + start_road + ',"' + cs_road + '",' + end_road + ']' + line.split("]")[1]
    return new_line


def get_start_pos(line):
    road_info = line.split("[")[1].split("]")[0]
    start_road = road_info.split(",")[0]
    return start_road


def get_end_pos(line):
    road_info = line.split("[")[1].split("]")[0]
    ll = len(road_info.split(","))
    return road_info.split(",")[ll - 1]


def get_flow_time(line):
    tmp = line.split(":")[13].split(",")[0]
    time = int(tmp)
    return time


def evcs_dynamic(region_num=4, cs_num=2):
    config.change_region_cs_num(region_num=region_num, cs_num=cs_num)
    revenue, revenue_cs, revenue_predict = simulations(time_interval)
    return {
        "revenue": revenue,
        "revenue_cs": revenue_cs,
        "revenue_predict": revenue_predict
    }


if __name__ == "__main__":
    get_part_road()
    init_intersection_info()
    evcs_dynamic(6, 5)
