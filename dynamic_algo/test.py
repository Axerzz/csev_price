
def get_end_pos(line):
    road_info = line.split("[")[1].split("]")[0]
    ll = len(road_info.split(","))
    return road_info.split(",")[ll-1]


if __name__ == "__main__":
    f = open("../flow.txt")
    line = f.readline()
    line = f.readline()
    # while line:
    line = line.strip("\n")
    print(line)
    print(get_end_pos(line))
    line = f.readline()
