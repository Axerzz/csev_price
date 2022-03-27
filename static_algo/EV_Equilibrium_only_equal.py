# -*- coding:utf-8 -*-
'''
@Project ：csev
@File    ：EV_Equilibrium_only_equal.py
@Author  ：xxuanZhu
@Date    ：2021/6/4 9:58
@Purpose : 底层均衡约束
'''

import copy
from random import shuffle
import sympy

import numpy as np
from scipy.optimize import minimize

from static_algo.config_exp import config


def trans(m):
    return list(zip(*m))


class EvEquilibrium(object):

    def fun_vehicles(self, args):
        v_i, dist_agent_to_cs_list, f_minus_i, price_list = args
        # zhx重构
        v = lambda x: sum([((config.w_p * price_list[k] + config.w_d * dist_agent_to_cs_list[k] +
                             config.w_qf * (v_i * x[k] + f_minus_i[k]) / config.cs_cap_vector[k]) * v_i * x[k])
                           for k in range(0, config.cs_num)])

        return v

    def con_strategy(self, x_min, x_max):  # 0, 1
        # 约束条件 分为eq 和ineq
        # eq表示 函数结果等于0 ； ineq 表示 表达式大于等于0

        cs_num = config.cs_num
        # zhx重构，先创建列表动态添加元素，最后转换为元组返回即可，不需要一开始就元组
        cons = [{'type': 'eq',
                 # 'fun': lambda x: x_max - x[0] - x[1] - x[2] - x[3]  # - x[4] - x[5] - x[6] - x[7] - x[8] - x[9]
                 'fun': lambda x: x_max - sum(x[i] for i in range(cs_num))
                 }]
        for i in range(0, cs_num):
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[i] - x_min})
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x_max - x[i]})

        return tuple(cons)

    def initiation(self, region_num, cs_num, prov_flag=True):
        """
        if prov_flag == False:
            use wuhan dist
        else:
            use nanjing dist
        """

        region_list = []

        if prov_flag:
            dist_vector = config.dist_vector
        else:
            dist_vector = config.wuhan_dist_vector

        vehicle_vector = config.vehicle_vector

        # 生成策略向量, (regin_num * cs_num)
        strategy_vector = []
        for i in range(config.region_num):
            s = np.random.rand(config.cs_num)
            strategy_vector.append(s)

        for i in range(region_num):
            region_list.append(int(i))
        return dist_vector, vehicle_vector, region_list, strategy_vector

    # 每个区域agent的best_response
    def agent_best_response(self, agent, region_list, dist_agent_to_cs_list, vehicle_vector, strategy_vector_list,
                            price_list, minimize_res_list):
        v_i = vehicle_vector[agent]  # i区域内的汽车数量
        f_minus_i = [0 for i in range(config.cs_num)]  # 除去i区域其他区域到充电桩们的车辆数量集合 size: cs_num
        for item in region_list:  # 区域i
            if item != agent:  # 不是当前的区域
                for cs in range(config.cs_num):
                    # TODO why sum{strategy_vector} is not 1
                    f_minus_i[cs] = f_minus_i[cs] + \
                                    vehicle_vector[item] * strategy_vector_list[item][cs]  # 除了i区域外其他区域派到充电站cs的车辆数量
        for cs in range(config.cs_num):
            f_minus_i[cs] = round(f_minus_i[cs], 0)
        args = (v_i, dist_agent_to_cs_list, f_minus_i, price_list)
        cons = self.con_strategy(0, 1)
        # 设置初始猜测值
        fun = self.fun_vehicles(args)

        # zhx重构
        list_x0 = tuple([0.1 for i in range(0, config.cs_num)])
        x0 = np.asarray(list_x0)  # , 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1
        res = minimize(fun, x0, method='SLSQP', constraints=cons)
        minimize_res_list.append(res.fun)
        # print("车辆最小化结果： ", res.fun, res.success, res.x)

        # zhx重构
        result_list = []
        for i in range(config.cs_num):
            result_list.append(round(res.x[i], 3))
        return tuple(result_list)

    def best_response_simulation(self, region_list, dist_vector_list, vehicle_vector, price_list,
                                 minimize_res_list, strategy_vector_list):
        epision = 0.000001
        num = 1
        round_num = num

        flag_new = copy.deepcopy(np.array(list(trans(strategy_vector_list))))  # 4 * 6 or (CS * region)
        new_region_list = copy.deepcopy(region_list)  # [0, 1, 2, 3, 4, 5]
        shuffle(new_region_list)  # 将序列的所有元素随机排序

        for agent in new_region_list:  # 区域i对桩
            dist_agent_to_cs_list = []  # 某个区域i到不同桩的距离 -> [2,4,5], 2是到桩1,4是到桩2,5是到桩3
            for cs in range(config.cs_num):
                dist = dist_vector_list[cs]  # [1,2,3,4,5]
                dist_agent_to_cs_list.append(dist[agent])  # [2]

            # 求当前区域i的最优策略
            strategy_vector_list[agent] = np.array(list(self.agent_best_response(agent, region_list,
                                                                                 dist_agent_to_cs_list,
                                                                                 vehicle_vector,
                                                                                 strategy_vector_list, price_list,
                                                                                 minimize_res_list)))
        # 把region*cs->cs*region
        strategy_vector_list = np.array(list(trans(strategy_vector_list)))

        # zhx重构：考虑在外面计算用循环计算较复杂的表达式
        while True:
            calculate_flag = False
            for i in range(config.cs_num):
                if np.linalg.norm(flag_new[i] - strategy_vector_list[i]) > epision:
                    calculate_flag = True
                    break
            if calculate_flag is True:  # 求二范数
                num = num + 1
                # cs * region->region * cs
                strategy_vector_list = np.array(list(trans(strategy_vector_list)))

                if num > 100:
                    print("没找到均衡！\n")
                    return strategy_vector_list, False

                # print("开始第", num, "轮更新：")
                flag_new = copy.deepcopy(np.array(list(trans(strategy_vector_list))))
                new_region_list = copy.deepcopy(region_list)
                shuffle(new_region_list)
                for agent in new_region_list:  # 区域i对桩
                    dist_agent_to_cs_list = []  # 某个区域i到不同桩的距离 -> [2,4,5], 2是到桩1,4是到桩2,5是到桩3
                    for cs in range(config.cs_num):
                        dist = dist_vector_list[cs]  # [1,2,3,4,5]
                        dist_agent_to_cs_list.append(dist[agent])  # [2]

                    # 求当前区域i的最优策略
                    # zhx重构，原来函数返回的是tuple，现在变成了list，所以要变回去
                    strategy_vector_list[agent] = tuple(self.agent_best_response(agent, region_list,
                                                                                 dist_agent_to_cs_list,
                                                                                 vehicle_vector,
                                                                                 strategy_vector_list, price_list,
                                                                                 minimize_res_list))
                strategy_vector_list = np.array(list(trans(strategy_vector_list)))
            else:
                break

        # print("均衡下的策略向量集合为：", trans(strategy_vector_list), "\n")
        return trans(strategy_vector_list), True

    def single_region_equilibrium_check(self, idx, strategy, price, dist):
        fjs = [0 for _ in range(config.cs_num)]
        for x in range(config.region_num):
            for y in range(config.cs_num):
                fjs[y] += strategy[x][y]
        non_zero_lambda, zero_lambda = -1, -1
        for j in range(config.cs_num):
            val = config.w_p * price[j] + config.w_d * dist[j][idx] + \
                  config.w_qf * (fjs[j] + strategy[idx][j]) / config.cs_cap_vector[j]
            if strategy[idx][j] == 0:
                if zero_lambda == -1:
                    zero_lambda = val
                zero_lambda = min(zero_lambda, val)
                if non_zero_lambda != -1 and non_zero_lambda - zero_lambda > 1e-4:
                    # print("<", zero_lambda, non_zero_lambda, ">")
                    return False, 1
            else:
                if non_zero_lambda == -1:
                    non_zero_lambda = val
                if np.absolute(val - non_zero_lambda) > 1e-4:
                    # print("<", val, non_zero_lambda, ">")
                    return False, 2
                if zero_lambda != -1 and non_zero_lambda - zero_lambda > 1e-4:
                    # print("<", zero_lambda, non_zero_lambda, ">")
                    return False, 3
        return True, 0

    def equilibrium_check(self, strategy, price, dist):
        fjs = [0 for _ in range(config.cs_num)]
        for x in range(config.region_num):
            for y in range(config.cs_num):
                fjs[y] += strategy[x][y]
        for i in range(config.region_num):
            non_zero_lambda, zero_lambda = -1, -1
            for j in range(config.cs_num):
                val = config.w_p * price[j] + config.w_d * dist[j][i] + \
                      config.w_qf * (fjs[j] + strategy[i][j]) / config.cs_cap_vector[j]
                if strategy[i][j] == 0:
                    if zero_lambda == -1:
                        zero_lambda = val
                    zero_lambda = min(zero_lambda, val)
                    if non_zero_lambda != -1 and non_zero_lambda - zero_lambda > 1e-4:
                        # print("1<", zero_lambda, non_zero_lambda, ">")
                        return False
                else:
                    if non_zero_lambda == -1:
                        non_zero_lambda = val
                    if np.absolute(val - non_zero_lambda) > 1e-4:
                        # print("2<", val, non_zero_lambda, ">")
                        return False
                    if zero_lambda != -1 and non_zero_lambda - zero_lambda > 1e-3:
                        # print("3<", zero_lambda, non_zero_lambda, ">")
                        return False
        return True

    def init_eqsys(self, system_dict, args, price, dist, vehicle_num):
        # init eqsys using mean value
        strategy = [[vehicle_num[i] / config.cs_num for _ in range(config.cs_num)] for i in range(config.region_num)]
        while True:
            strategy_bak = copy.deepcopy(strategy)
            for i in range(config.region_num):
                system_dict_bak = copy.deepcopy(system_dict)
                # init system
                for exp in system_dict_bak.items():
                    system_dict_bak[exp[0]] = sympy.symbols(exp[0])
                    if exp[0][0] == "f" and exp[0][1:].split("-")[0] != str(i):
                        system_dict_bak[exp[0]] -= strategy[int(exp[0][1:].split("-")[0])][
                            int(exp[0][1:].split("-")[1])]
                system_dict_bak["n" + str(i)] = system_dict["n" + str(i)]
                # cal and sort the cost for each cs according to region i
                cost_dict = {}
                for j in range(config.cs_num):
                    cost_dict["f" + str(i) + "-" + str(j)] = \
                        config.w_p * price[j] + config.w_d * dist[j][i] + \
                        config.w_qf * (sum([strategy[k][j] for k in range(config.region_num)])
                                       - strategy[i][j]) / config.cs_cap_vector[j]
                cost_list = sorted(cost_dict.items(), key=lambda c: c[1])
                level, cs_cap = 0, config.cs_num
                for hi in range(1, config.cs_num):
                    for lo in range(hi):
                        tmp_idx = int(cost_list[lo][0][1:].split("-")[1])
                        level += (float(cost_list[hi][1]) - float(cost_list[lo][1])) * \
                                 config.cs_cap_vector[tmp_idx] / (2 * config.w_qf)
                    if level >= vehicle_num[i]:
                        # print(hi, level-2*vehicle_num[i])
                        cs_cap = hi
                        break
                    level = 0
                for cs in range(cs_cap):
                    system_dict_bak[cost_list[cs][0]] = system_dict[cost_list[cs][0]]
                solution = sympy.solve(system_dict_bak.values(), args)
                tmp = [0.0 for _ in range(config.cs_num)]
                for exp in solution.items():
                    exp_id = str(exp[0])
                    if exp_id[0] != "f":
                        continue
                    region_idx, cs_idx = int(exp_id[1:].split("-")[0]), int(exp_id[1:].split("-")[1])
                    if region_idx != i:
                        continue
                    base_value = float(exp[1])
                    if base_value < 0:
                        if base_value < -1e-6:
                            print(base_value)
                            assert False
                        else:
                            tmp[cs_idx] = 0
                    else:
                        tmp[cs_idx] = base_value
                for j in range(config.cs_num):
                    strategy[i][j] = tmp[j]
                # eq, code = self.single_region_equilibrium_check(i, strategy, price, dist)
                # print("region: %d, code: %d" % (i, code))
                # for r in range(config.region_num):
                #     print(strategy[r])
            err = np.linalg.norm(np.array(strategy).astype(float) - np.array(strategy_bak).astype(float))
            if err < 1e-4:
                break
            # else:
            #     print("err: ", err)
        assert self.equilibrium_check(strategy, price, dist)
        return strategy


evEquilibrium = EvEquilibrium()

if __name__ == "__main__":
    config.change_region_cs_num(region_num=8, cs_num=6)
    dist, vehicle_num, region, strategy_vector = evEquilibrium.initiation(8, 6, True)
    a = [[70.0, 0.0, 0.0, 0.0, 0.0, 0.0],
         [0.0, 130.0, 0.0, 0.0, 0.0, 0.0],
         [45.686360286632286, 0.0, 122.31363971336772, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 32.0, 0.0, 0.0],
         [186.37149754332023, 2.422231383333333e-06, 0.0, 0.0, 171.6285000344484, 0.0],
         [0.0, 44.54027905355332, 0.0, 0.0, 0.0, 163.45972094644668],
         [140.0, 0.0, 0.0, 0.0, 0.0, 0.0],
         [80.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    eveq = evEquilibrium.equilibrium_check(a,
                                           [45.7539682539683, 55.0842203548085, 61.9282389214681,
                                            61.9106255541785, 62.3976447902010, 61.3589697745215],
                                           dist)
