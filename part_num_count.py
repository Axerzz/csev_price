import numpy as np

part_road = [[] for i in range(6)]
part_ev_count = [0 for j in range(6)]


def get_part_ev_count():
    f = open("flow.txt")
    line = f.readline()
    while line:
        line = line.strip("\n")
        road1 = line.split("[")[1].split(",")[0]
        # print(road1)
        for i in range(6):
            if road1 in part_road[i]:
                part_ev_count[i] = part_ev_count[i]+1
        line = f.readline()
    f.close()


def get_part_road():
    for i in range(1, 7):
        s = 'partition_road/partition' + str(i) + '.txt'
        f = open(s)
        line = f.readline()
        while line:
            line = line.strip("\n")
            part_road[i - 1].append(line)
            line = f.readline()
        f.close()


if __name__ == "__main__":
    get_part_road()
    get_part_ev_count()
    w = open("part_ev_count.txt", "w+")
    for i in range(1,7):
        print("partition "+str(i)+": "+str(part_ev_count[i-1]),file=w)
    w.close()
