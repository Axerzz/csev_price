import numpy as np

road_count = {}
vehicle_info = {}


def road_counting(line_string):
    road_info = line_string.split("[")[1].split("]")[0]
    single_road = road_info.split(",")
    # print(single_road)
    for i in range(len(single_road)):
        if single_road[i] in road_count:
            road_count[single_road[i]] += 1
        else:
            road_count[single_road[i]] = 1
    # print(road_count)


def time_extract(line_string):
    tmp = line_string.split(":")[13].split(",")[0]
    time = int(tmp)
    vehicle_info[line_string] = time


if __name__ == "__main__":
    f = open("flow.txt")
    line = f.readline()
    while line:
        line = line.strip("\n")
        road_counting(line)
        time_extract(line)
        line = f.readline()

    sorted_road_count = sorted(road_count.items(), key=lambda d: d[1], reverse=True)
    w_road_count = open("road_count.txt", 'w+')
    for i in range(20):
        print(sorted_road_count[i][0] + ":" + str(sorted_road_count[i][1]), file=w_road_count)

    sorted_vehicle_info = sorted(vehicle_info.items(), key=lambda d: d[1])
    w_vehicle_info = open("flow_sorted.txt", 'w+')
    print("[", file=w_vehicle_info)
    for i in range(len(vehicle_info)):
        print(sorted_vehicle_info[i][0], file=w_vehicle_info)
    print("]", file=w_vehicle_info)
    f.close()
    w_road_count.close()
    w_vehicle_info.close()
