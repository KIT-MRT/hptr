import h5py
import numpy as np
import matplotlib.pyplot as plt

DATA_ROOT_DIR = ""

with h5py.File(DATA_ROOT_DIR + "training.h5", "r") as h5_file:
    # file ids: 0, 1, 2, ...
    for key in h5_file.keys():
        scenario_goup = h5_file[key]
        # 5804b6eb02db5734_60, 2b078b7743b55dc5_20
        if scenario_goup.attrs["scenario_id"] == "2b078b7743b55dc5_20":
            break

    # scenario_goup = h5_file["150"]

    print(scenario_goup.attrs["scenario_id"])
    agent = scenario_goup["agent"]
    agent_pos = agent["pos"]  # [n_step, N_AGENT_MAX, 3], float32
    agent_valid = agent["valid"]  # [n_step, N_AGENT_MAX], bool
    agent_type = agent[
        "type"
    ]  # [N_AGENT_MAX, 3], bool, one hot [Vehicle=0, Pedestrian=1, Cyclist=2]

    map = scenario_goup["map"]
    map_pos = map["pos"]  # [N_PL_MAX, 20, 3]
    map_valid = map["valid"]  # [N_PL_MAX, 20], bool
    map_type = map["type"]  # [N_PL_MAX], int, >= 0

    route_data = scenario_goup["route"]
    route_pos = route_data["pos"]
    route_valid = route_data["valid"]
    route_type = route_data["type"]
    route_goal = route_data["goal"]

    tl_lane = scenario_goup["tl_lane"]
    tl_lane_valid = tl_lane["valid"]

    tl_stop = scenario_goup["tl_stop"]
    tl_stop_pos = tl_stop["pos"]
    tl_stop_valid = tl_stop["valid"]

    for pl, valid, type in zip(map_pos, map_valid, map_type):
        pl = pl[valid]
        # Crosswalk, walkways, boundaries, carpark_area
        if type[2] or type[3] or type[4] or type[5]:
            plt.plot(pl[:, 0], pl[:, 1], "-", c="black", linewidth=2)
        # Stop line
        elif type[1]:
            plt.plot(pl[:, 0], pl[:, 1], "-", c="red", zorder=-10, linewidth=2)
        # Centerline
        elif type[6]:
            plt.plot(pl[:, 0], pl[:, 1], "-", c="green", zorder=-10)
        # Intersection
        elif type[0]:
            plt.plot(pl[:, 0], pl[:, 1], "-", c="gray", zorder=-10)

    for pl, valid, type in zip(route_pos, route_valid, route_type):
        pl = pl[valid]
        # Route
        if type[7]:
            plt.plot(pl[:, 0], pl[:, 1], "-", c="blue", zorder=-10)

    for pl, valid in zip(tl_stop_pos, tl_stop_valid):
        pl = pl[valid]
        plt.plot(pl[:, 0], pl[:, 1], "o", c="red", zorder=-10, markersize=7)

    for step in range(agent_pos.shape[0]):
        for pl, valid, type in zip(agent_pos[step], agent_valid[step], agent_type):
            pl = pl[valid]
            plt.plot(pl[:, 0], pl[:, 1], "o", c="orange", markersize=2)

    plt.plot(route_goal[0], route_goal[1], "x", c="blue", markersize=10)
    goal_x, goal_y, goal_yaw = route_goal
    yaw_length = 10  # Length of the yaw vector
    yaw_x = goal_x + yaw_length * np.cos(goal_yaw)
    yaw_y = goal_y + yaw_length * np.sin(goal_yaw)
    plt.arrow(
        goal_x,
        goal_y,
        yaw_x - goal_x,
        yaw_y - goal_y,
        head_width=1,
        head_length=2,
        fc="blue",
        ec="blue",
    )

    plt.show()
