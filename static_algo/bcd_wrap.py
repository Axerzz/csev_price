#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project:   csev
@File:      bcd_wrap.py
@Author:    bai
@Email:     wenchao.bai@qq.com
@Date:      2021/11/4 10:55
@Purpose:   impl. for BCD algo.
"""

import time
import copy
import sympy
import numpy as np
import logging

from static_algo.config_exp import config
from static_algo.EV_Equilibrium_only_equal import EvEquilibrium
from static_algo import smooth

evEquilibrium = EvEquilibrium()
logging.basicConfig(level=logging.INFO)


def optimize_p(cs_idx, q, left, right, price):
    exp = 0
    for j in range(config.cs_num):
        if j != cs_idx:
            exp += q[j] * (price[j] - config.cost[j])
        else:
            exp += q[j] * (sympy.symbols("jp") - config.cost[j])
    ans = {"p": -1, "r": -1}
    djp = sympy.diff(exp, sympy.symbols("jp"))
    roots = sympy.solve(djp, sympy.symbols("jp"))
    left_val = eval(str(exp).replace("jp", str(left)))
    right_val = eval(str(exp).replace("jp", str(right)))
    if not roots:
        if left_val < right_val:
            ans["p"] = right
            ans["r"] = right_val
            return ans
        if right_val < left_val:
            ans["p"] = left
            ans["r"] = left_val
            return ans
        logging.info(
            "expression of revenue is constant: " + str(q[cs_idx]) + ", left=" + str(left) + ", right=" + str(right))
        if left <= price[cs_idx] <= right:
            ans["p"] = price[cs_idx]
            ans["r"] = left_val
            return ans
        if np.absolute(left - price[cs_idx]) <= np.absolute(right - price[cs_idx]):
            ans["p"] = left
            ans["r"] = left_val
        else:
            ans["p"] = right
            ans["r"] = right_val
        return ans
    if right_val > left_val:
        ans["p"] = right
        ans["r"] = right_val
    else:
        ans["p"] = left
        ans["r"] = left_val
    if str(roots[0])[-1] == "I":
        return ans
    else:
        for i in range(len(roots)):
            val = roots[i]
            if left >= val or right <= val:
                continue
            tmp_r = eval(str(exp).replace("jp", str(val)))
            if tmp_r > ans["r"]:
                ans["p"] = val
                ans["r"] = tmp_r
                logging.info("use non-boundary value, p=" + str(val) + ", r=" + str(tmp_r) +
                             ", left=" + str(left) + ", right=" + str(right))
    return ans


def nxt_jp_revenue(cs_idx, system_dict, args, jump_base, neg_list, price, dist, system_dict_bak):
    solution = sympy.solve(system_dict.values(), args)
    pos, neg, neg_exp, stop_flag = {}, {}, {}, False

    # represent Q_i using jp
    q = [0 for _ in range(config.cs_num)]
    for j in range(config.cs_num):
        for i in range(config.region_num):
            q[j] += solution[sympy.symbols("f" + str(i) + "-" + str(j))]

    for exp in solution.items():
        if str(exp[0])[0] != "f" or str(exp[0]) in neg_list:
            continue
        base_value = eval(str(exp[1]).replace("jp", str(jump_base)))
        if eval(str(exp[1]).replace("jp", str(jump_base + 1))) < base_value <= 0:
            logging.warning("illegal non-positive expression exists: " + str(exp) + "; value: " + str(base_value))
        elif eval(str(exp[1]).replace("jp", str(jump_base + 1))) < base_value:
            pos[str(exp[0])] = sympy.solve(exp[1], args[-1:])[0]

    for neg_id in neg_list:
        x, y = int(neg_id[1:].split("-")[0]), int(neg_id[1:].split("-")[1])
        exp = -1 * config.w_d * dist[y][x] + solution[sympy.symbols("lambda" + str(x))]
        exp_ori = system_dict_bak["f" + str(x) + "-" + str(y)]
        if cs_idx == int(y):
            exp -= config.w_p * args[-1]
        else:
            exp -= config.w_p * price[y]
        for i in range(config.region_num):
            exp -= config.w_qf * solution[sympy.symbols("f" + str(i) + "-" + str(y))] / config.cs_cap_vector[y]
        exp *= config.cs_cap_vector[y] / config.w_qf
        neg_exp[neg_id] = exp_ori
        base_value = eval(str(exp).replace("jp", str(jump_base)))
        if 0 <= base_value < eval(str(exp).replace("jp", str(jump_base + 1))):
            logging.warning("illegal non-negative expression exists: " + str(exp) + "; value: " + str(base_value))
        elif eval(str(exp).replace("jp", str(jump_base + 1))) > base_value:
            neg[neg_id] = sympy.solve(exp, args[-1:])[0]

    pos_li = sorted(pos.items(), key=lambda p: p[1])
    neg_li = sorted(neg.items(), key=lambda n: n[1])

    pos_head, neg_head, nxt_jp = config.P_MAX, config.P_MAX, config.P_MAX
    if len(pos_li) != 0:
        pos_head = pos_li[0][1]
    if len(neg_li) != 0:
        neg_head = neg_li[0][1]
    if pos_head >= config.P_MAX and neg_head >= config.P_MAX:
        stop_flag = True
        ans = optimize_p(cs_idx, q, jump_base, config.P_MAX, price)
        return stop_flag, system_dict, config.P_MAX, neg_list, ans

    if pos_head <= neg_head:
        nxt_jp, idx = pos_head, 0
        while idx < len(pos_li) and nxt_jp == pos_li[idx][1]:
            system_dict[str(pos_li[idx][0])] = sympy.symbols(pos_li[idx][0])
            neg_list.append(str(pos_li[idx][0]))
            idx += 1
    if neg_head <= pos_head:
        nxt_jp, idx = neg_head, 0
        while idx < len(neg_li) and nxt_jp == neg_li[idx][1]:
            system_dict[str(neg_li[idx][0])] = neg_exp[neg_li[idx][0]]
            neg_list.remove(str(neg_li[idx][0]))
            idx += 1

    ans = optimize_p(cs_idx, q, jump_base, nxt_jp, price)

    return stop_flag, system_dict, nxt_jp, neg_list, ans


def optimize_single_cs(cs_idx, system_dict, args, price, dist, strategy):
    system_dict_bak = copy.deepcopy(system_dict)
    neg_list = []
    for i in range(config.region_num):
        for j in range(config.cs_num):
            if strategy[i][j] < 1e-4:
                system_dict["f" + str(i) + "-" + str(j)] = sympy.symbols("f" + str(i) + "-" + str(j))
                neg_list.append("f" + str(i) + "-" + str(j))
    # check if any negative f_{ij} exists
    solution = sympy.solve(system_dict.values(), args)
    for exp in solution.items():
        exp_id = str(exp[0])
        if exp_id[0] != "f":
            continue
        base_value = eval(str(exp[1]).replace("jp", str(price[cs_idx])))
        if base_value <= 0 and str(exp[0]) not in neg_list:
            logging.warning("negative f_{ij} exists in the initialized equation system! " + str(base_value))
            system_dict[exp_id] = exp[0]
            neg_list.append(str(exp[0]))

    stop_flag, nxt_jp, jps, ans_list = False, config.P_MIN, [config.P_MIN], []
    orp_dict = {}
    while True:
        stop_flag, system_dict, nxt_jp, neg_list, ans = \
            nxt_jp_revenue(cs_idx, system_dict, args, nxt_jp, neg_list, price, dist, system_dict_bak)
        logging.info("next jump point: " + str(nxt_jp) + " p: " + str(ans["p"]) + " r: " + str(ans["r"]))
        orp_dict[ans["p"]] = ans["r"]
        jps.append(nxt_jp)
        if stop_flag:
            break
    orp_list = sorted(orp_dict.items(), key=lambda o: o[1], reverse=True)
    o_p, o_r, abs_dist, orp_idx = orp_list[0][0], round(orp_list[0][1], 2), np.absolute(
        price[cs_idx] - orp_list[0][0]), 1
    while orp_idx < len(orp_list) and round(orp_list[orp_idx][1], 2) == o_r:
        if np.absolute(price[cs_idx] - orp_list[orp_idx][0]) < abs_dist:
            o_p = orp_list[orp_idx][0]
        orp_idx += 1

    return jps, o_p, o_r


def optimize(vehicle_num, price, dist):
    """
    define symbols
    """
    f_symbols = [[sympy.symbols("f" + str(i) + "-" + str(j)) for j in range(config.cs_num)] for i in
                 range(config.region_num)]
    q_symbols = [sympy.symbols("q" + str(j)) for j in range(config.cs_num)]
    lambda_symbols = [sympy.symbols("lambda" + str(i)) for i in range(config.region_num)]

    """  
    define equation system
    """
    jp = sympy.symbols("jp")  # jump point (price of crt cs)
    equation_dict = {}
    # use f to represent q
    for j in range(config.cs_num):
        q_symbols[j] = 0
        for i in range(config.region_num):
            q_symbols[j] += f_symbols[i][j]

    # n-constraint equations
    for i in range(config.region_num):
        equation_dict["n" + str(i)] = -vehicle_num[i]
        for j in range(config.cs_num):
            equation_dict["n" + str(i)] += f_symbols[i][j]

    # equilibrium equations
    price_tmp = copy.deepcopy(price)
    price_tmp[0] = config.P_MIN
    systems = [{} for _ in range(config.cs_num + 1)]
    for crt_cs in range(config.cs_num + 1):
        for i in range(config.region_num):
            for j in range(config.cs_num):
                if crt_cs == j:
                    equation_dict["f" + str(i) + "-" + str(j)] = \
                        config.w_p * jp + \
                        config.w_qf * q_symbols[j] / config.cs_cap_vector[j] + \
                        config.w_d * dist[j][i] + \
                        config.w_qf * f_symbols[i][j] / config.cs_cap_vector[j] - \
                        lambda_symbols[i]
                else:
                    equation_dict["f" + str(i) + "-" + str(j)] = \
                        config.w_p * price_tmp[j] + \
                        config.w_qf * q_symbols[j] / config.cs_cap_vector[j] + \
                        config.w_d * dist[j][i] + \
                        config.w_qf * f_symbols[i][j] / config.cs_cap_vector[j] - \
                        lambda_symbols[i]
        systems[crt_cs] = copy.deepcopy(equation_dict)

    # arg list
    arg_f = [f_symbols[i][j] for j in range(config.cs_num) for i in range(config.region_num)]
    arg_lambda = [lambda_symbols[i] for i in range(config.region_num)]
    arg_jp = [jp]
    args = arg_f + arg_lambda + arg_jp

    # init strategy vector
    strategy = evEquilibrium.init_eqsys(systems[config.cs_num], args[:-1], price_tmp, dist, vehicle_num)
    revenue, break_flag, count = -1, False, 0
    while True:
        past_price = copy.deepcopy(price)
        for j in range(config.cs_num):
            print(">> CS %d, round %d" % (j, count))
            _, price[j], revenue = optimize_single_cs(j, copy.deepcopy(systems[j]), args, price_tmp, dist, strategy)
            logging.info(str(j) + ": price: " + str(price[j]) + " prices: " + str(price) + " revenue: " + str(revenue))
            # update equation systems
            price_tmp = copy.deepcopy(price)
            price_tmp[(j + 1) % config.cs_num] = config.P_MIN
            for x in range(config.region_num):
                for y in range(config.cs_num):
                    systems[config.cs_num]["f" + str(x) + "-" + str(y)] = \
                        config.w_p * price_tmp[y] + \
                        config.w_qf * q_symbols[y] / config.cs_cap_vector[y] + \
                        config.w_d * dist[y][x] + \
                        config.w_qf * f_symbols[x][y] / config.cs_cap_vector[y] - \
                        lambda_symbols[x]
                    if (j + 1) % config.cs_num != y:
                        systems[(j + 1) % config.cs_num]["f" + str(x) + "-" + str(y)] = \
                            config.w_p * price_tmp[y] + \
                            config.w_qf * q_symbols[y] / config.cs_cap_vector[y] + \
                            config.w_d * dist[y][x] + \
                            config.w_qf * f_symbols[x][y] / config.cs_cap_vector[y] - \
                            lambda_symbols[x]
            strategy = evEquilibrium.init_eqsys(systems[config.cs_num], args[:-1], price_tmp, dist, vehicle_num)
        if np.linalg.norm(np.array(price).astype(float) - np.array(past_price).astype(float)) < 1e-6 or count > 50:
            break
        count += 1

    return revenue, price


def evcs_bcd(region_num=6, cs_num=4, prov_flag=True):
    print(">> start BCD! region=%d, cs=%d, prov_flag=%s" % (region_num, cs_num, prov_flag))
    begin_time = time.time()

    # init price
    price = smooth.smooth_algo_noV(region_num, cs_num, prov_flag)
    print("smooth_price:",price)

    # read config according to meta-parameters
    config.change_region_cs_num(region_num=region_num, cs_num=cs_num)
    dist, vehicle_num, region, strategy_vector = \
        evEquilibrium.initiation(region_num, cs_num, prov_flag)

    revenue, opt_price = optimize(vehicle_num, price, dist)

    end_time = time.time()
    print("<< end BCD! region=%d, cs=%d, prov_flag=%s" % (region_num, cs_num, prov_flag))

    return {
        "profit": revenue,
        "time": end_time - begin_time,
        "price": opt_price
    }


if __name__ == "__main__":
    ans_dict = evcs_bcd(region_num=6, cs_num=5, prov_flag=True)
    print("optimal price:", ans_dict["price"])
    print("optimal revenue:", ans_dict["profit"])
    print("time:", ans_dict["time"])
