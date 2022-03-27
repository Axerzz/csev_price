#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2022/3/24
# @Author : lyw
# @Version：V 0.1
# @File : cal_best_cs.py
# @desc : calculate best cs for ev


import math

intersection_info = []

cs_info = [[31.851441415, 117.1740803],
           [31.85064675, 117.1153743],
           [31.83534871, 117.1341873],
           [31.81164271, 117.1153179],
           [31.82085063, 117.1759037]]

cs_intersection = [["22_12", "22_13", "33_1_0", "33_2_2"],
                   ["22_3", "22_4", "36_1_0", "36_2_2"],
                   ["17_8", "1_6", "37_3_3", "37_4_1"],
                   ["6_3", "6_4", "35_1_0", "35_2_2"],
                   ["4_11", "4_12", "34_1_0", "34_2_2"]]

EARTH_REDIUS = 6378.137


def rad(d):
    return (d * math.pi) / 180.0


def getDistance(lat1, lng1, lat2, lng2):
    radLat1 = rad(lat1)
    radLat2 = rad(lat2)
    a = radLat1 - radLat2
    b = rad(lng1) - rad(lng2)
    s = 2 * math.asin(
        math.sqrt(math.pow(math.sin(a / 2), 2) + math.cos(radLat1) * math.cos(radLat2) * math.pow(math.sin(b / 2), 2)))
    s = s * EARTH_REDIUS
    return s


def init_intersection_info():
    f = open("../intersection_pos.txt")
    line = f.readline()
    while line:
        line = line.strip("\n")
        intersection_info.append(line.split("\t"))
        line = f.readline()


def get_rightAngle_dist(x1, y1, x2, y2):
    dis = getDistance(x1, y1, x2, y1) + getDistance(x2, y1, x2, y2)
    dis = dis * 10
    return dis


def get_pos_from_intersection(intersection):
    latitude = 0.0
    longitude = 0.0
    for i in range(len(intersection_info)):
        if intersection_info[i][0] == intersection:
            latitude = float(intersection_info[i][1])
            longitude = float(intersection_info[i][2])
            break
    return latitude, longitude


def get_best_cs_road(start_road, best_cs):
    s_l = len(start_road)
    intersection_start = "intersection" + start_road[5:s_l - 3]
    latitude, longitude = get_pos_from_intersection(intersection_start)

    intersection_best_cs_1 = "intersection_" + cs_intersection[best_cs][0]
    latitude_1, longitude_1 = get_pos_from_intersection(intersection_best_cs_1)
    intersection_best_cs_2 = "intersection_" + cs_intersection[best_cs][1]
    latitude_2, longitude_2 = get_pos_from_intersection(intersection_best_cs_2)

    dis1 = get_rightAngle_dist(latitude, longitude, latitude_1, longitude_1)
    dis2 = get_rightAngle_dist(latitude, longitude, latitude_2, longitude_2)
    if dis1 < dis2:
        return "road_" + cs_intersection[best_cs][2]
    else:
        return "road_" + cs_intersection[best_cs][3]


def get_best_cs(start_road, end_road, price, cs_in_queue):
    ll = len(start_road)
    intersection_start = "intersection" + start_road[5:ll - 3]
    ll = len(end_road)
    intersection_end = "intersection" + end_road[5:ll - 3]
    latitude, longitude = get_pos_from_intersection(intersection_start)
    latitude_end, longitude_end = get_pos_from_intersection(intersection_end)
    best_dist = 0
    minx = 1e9
    best_cs = 0
    best_dist1 = 0
    best_dist2 = 0
    for i in range(5):
        dist1 = get_rightAngle_dist(latitude, longitude, cs_info[i][0], cs_info[i][1])
        dist2 = get_rightAngle_dist(latitude_end, longitude_end, cs_info[i][0], cs_info[i][1])
        tmp = 0.6 * price[i] + 0.1 * cs_in_queue[i] + 0.3 * (dist1 + dist2)
        if tmp < minx:
            best_cs = i
            best_dist = dist1 + dist2
            minx = tmp
            best_dist1 = dist1
            best_dist2 = dist2
    return best_cs, best_dist, intersection_start, best_dist1, best_dist2


if __name__ == "__main__":
    init_intersection_info()
    print(get_best_cs_road('"road_1_1_1"',1))
