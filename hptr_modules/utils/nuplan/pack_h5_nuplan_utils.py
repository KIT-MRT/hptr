# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
import logging
import os
import math
import hydra
import tempfile
import numpy as np
from dataclasses import dataclass
from os.path import join
from typing import Union, List, Set

import nuplan
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.nuplan_map.utils import (
    get_distance_between_map_object_and_point,
)
from nuplan.planning.training.preprocessing.feature_builders.vector_builder_utils import (
    prune_route_by_connectivity,
)
from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.planning.script.builders.scenario_building_builder import (
    build_scenario_builder,
)
from nuplan.planning.script.builders.scenario_filter_builder import (
    build_scenario_filter,
)
from nuplan.planning.script.utils import set_up_common_builder

NUPLAN_PACKAGE_PATH = os.path.dirname(nuplan.__file__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_nuplan_scenarios(
    data_root, map_root, logs: Union[list, None] = None, builder="nuplan_mini"
):
    """
    Getting scenarios. You could use your parameters to get a bunch of scenarios
    :param data_root: path contains .db files, like /nuplan-v1.1/splits/mini
    :param map_root: path to map
    :param logs: a list of logs, like ['2021.07.16.20.45.29_veh-35_01095_01486']. If none, load all files in data_root
    :param builder: builder file, we use the default nuplan builder file
    :return:
    """
    nuplan_package_path = NUPLAN_PACKAGE_PATH
    logs = logs or [file for file in os.listdir(data_root)]
    log_string = ""
    for log in logs:
        if log[-3:] == ".db":
            log = log[:-3]
        log_string += log
        log_string += ","
    log_string = log_string[:-1]

    dataset_parameters = [
        # builder setting
        "scenario_builder={}".format(builder),
        "scenario_builder.scenario_mapping.subsample_ratio_override=0.5",  # 1 = 20Hz -> 0.5 = 10Hz
        "scenario_builder.data_root={}".format(data_root),
        "scenario_builder.map_root={}".format(map_root),
        # filter
        "scenario_filter=all_scenarios",  # simulate only one log
        "scenario_filter.remove_invalid_goals=true",
        "scenario_filter.shuffle=true",
        "scenario_filter.log_names=[{}]".format(log_string),
        # "scenario_filter.scenario_types={}".format(all_scenario_types),
        # "scenario_filter.scenario_tokens=[]",
        # "scenario_filter.map_names=[]",
        # "scenario_filter.num_scenarios_per_type=1",
        # "scenario_filter.limit_total_scenarios=1000",
        # "scenario_filter.expand_scenarios=true",
        # "scenario_filter.limit_scenarios_per_type=10",  # use 10 scenarios per scenario type
        "scenario_filter.timestamp_threshold_s=20",  # minial scenario duration (s)
    ]

    base_config_path = os.path.join(nuplan_package_path, "planning", "script")
    simulation_hydra_paths = construct_simulation_hydra_paths(base_config_path)

    # Initialize configuration management system
    hydra.core.global_hydra.GlobalHydra.instance().clear()  # reinitialize hydra if already initialized
    hydra.initialize_config_dir(config_dir=simulation_hydra_paths.config_path)

    save_dir = tempfile.mkdtemp()
    ego_controller = "perfect_tracking_controller"  # [log_play_back_controller, perfect_tracking_controller]
    observation = "box_observation"  # [box_observation, idm_agents_observation, lidar_pc_observation]

    # Compose the configuration
    overrides = [
        f"group={save_dir}",
        "worker=ray_distributed",  # sequential
        f"ego_controller={ego_controller}",
        f"observation={observation}",
        f"hydra.searchpath=[{simulation_hydra_paths.common_dir}, {simulation_hydra_paths.experiment_dir}]",
        "output_dir=${group}/${experiment}",
        "metric_dir=${group}/${experiment}",
        *dataset_parameters,
    ]
    overrides.extend(
        [
            f"job_name=planner_tutorial",
            "experiment=${experiment_name}/${job_name}",
            f"experiment_name=planner_tutorial",
        ]
    )

    # get config
    cfg = hydra.compose(
        config_name=simulation_hydra_paths.config_name, overrides=overrides
    )

    profiler_name = "building_simulation"
    common_builder = set_up_common_builder(cfg=cfg, profiler_name=profiler_name)

    # Build scenario builder
    scenario_builder = build_scenario_builder(cfg=cfg)
    scenario_filter = build_scenario_filter(cfg.scenario_filter)

    # get scenarios from database
    return scenario_builder.get_scenarios(scenario_filter, common_builder.worker)


def construct_simulation_hydra_paths(base_config_path: str):
    """
    Specifies relative paths to simulation configs to pass to hydra to declutter tutorial.
    :param base_config_path: Base config path.
    :return: Hydra config path.
    """
    common_dir = "file://" + join(base_config_path, "config", "common")
    config_name = "default_simulation"
    config_path = join(base_config_path, "config", "simulation")
    experiment_dir = "file://" + join(base_config_path, "experiments")
    return HydraConfigPaths(common_dir, config_name, config_path, experiment_dir)


@dataclass
class HydraConfigPaths:
    """
    Stores relative hydra paths to declutter tutorial.
    """

    common_dir: str
    config_name: str
    config_path: str
    experiment_dir: str


def compute_angular_velocity(initial_heading, final_heading, dt):
    """
    Calculate the angular velocity between two headings given in radians.

    Parameters:
    initial_heading (float): The initial heading in radians.
    final_heading (float): The final heading in radians.
    dt (float): The time interval between the two headings in seconds.

    Returns:
    float: The angular velocity in radians per second.
    """

    # Calculate the difference in headings
    delta_heading = final_heading - initial_heading

    # Adjust the delta_heading to be in the range (-π, π]
    delta_heading = (delta_heading + math.pi) % (2 * math.pi) - math.pi

    # Compute the angular velocity
    angular_vel = delta_heading / dt

    return angular_vel


def nuplan_to_centered_vector(vector, nuplan_center=(0, 0)):
    "All vec in nuplan should be centered in (0,0) to avoid numerical explosion"
    vector = np.array(vector)
    vector -= np.asarray(nuplan_center)
    return vector


def resample_polyline(path, polyline, sample_distance=0.5):
    length = path.length
    num_samples_final = max(length // sample_distance, 1)
    num_samples_path = len(path.discrete_path)
    if num_samples_path <= num_samples_final:
        return polyline
    else:
        sample_rate = int(num_samples_path // num_samples_final)
    return polyline[::sample_rate]


def extract_centerline(map_obj, nuplan_center, resampling=False, sample_distance=0.5):
    path = map_obj.baseline_path.discrete_path
    points = np.array(
        [nuplan_to_centered_vector([pose.x, pose.y], nuplan_center) for pose in path]
    )
    if resampling:
        points = resample_polyline(map_obj.baseline_path, points, sample_distance)
    return points


def get_points_from_boundary(boundary, center, resampling=False, sample_distance=0.5):
    path = boundary.discrete_path
    points = [(pose.x, pose.y) for pose in path]
    points = nuplan_to_centered_vector(points, center)
    if resampling:
        points = resample_polyline(boundary, points, sample_distance)
    return points


def mock_2d_to_3d_points(points):
    if len(points) == 0:
        return []
    # Convert input to numpy array
    points = np.array(points)
    # If points is a 1D array (single point), reshape it to 2D
    if points.ndim == 1:
        points = points.reshape(1, -1)
    # Add a third coordinate (z=0) to each point
    return np.hstack([points, np.zeros((points.shape[0], 1))])


def set_light_position(map_api, lane_id, center, target_position=8):
    lane = map_api.get_map_object(str(lane_id), SemanticMapLayer.LANE_CONNECTOR)
    assert lane is not None, "Can not find lane: {}".format(lane_id)
    path = lane.baseline_path.discrete_path
    acc_length = 0
    point = [path[0].x, path[0].y]
    for k, point in enumerate(path[1:], start=1):
        previous_p = path[k - 1]
        acc_length += np.linalg.norm([point.x - previous_p.x, point.y - previous_p.y])
        if acc_length > target_position:
            break
    return [point.x - center[0], point.y - center[1]]


def parse_object_state(obj_state, nuplan_center):
    ret = {}
    ret["position"] = nuplan_to_centered_vector(
        [obj_state.center.x, obj_state.center.y], nuplan_center
    )
    ret["heading"] = obj_state.center.heading
    ret["velocity"] = nuplan_to_centered_vector(
        [obj_state.velocity.x, obj_state.velocity.y]
    )
    ret["valid"] = 1
    ret["length"] = obj_state.box.length
    ret["width"] = obj_state.box.width
    ret["height"] = obj_state.box.height
    return ret


def parse_ego_vehicle_state(state, nuplan_center):
    center = nuplan_center
    ret = {}
    ret["position"] = nuplan_to_centered_vector(
        [state.waypoint.x, state.waypoint.y], center
    )
    ret["heading"] = state.waypoint.heading
    ret["velocity"] = nuplan_to_centered_vector(
        [state.agent.velocity.x, state.agent.velocity.y]
    )
    ret["angular_velocity"] = state.dynamic_car_state.angular_velocity
    ret["valid"] = 1
    ret["length"] = state.agent.box.length
    ret["width"] = state.agent.box.width
    ret["height"] = state.agent.box.height
    return ret


def parse_ego_vehicle_state_trajectory(
    scenario, nuplan_center, start_iter=None, stop_iter=None
):
    if start_iter is None:
        data = [
            parse_ego_vehicle_state(
                scenario.get_ego_state_at_iteration(i), nuplan_center
            )
            for i in range(scenario.get_number_of_iterations())
        ]
    else:
        data = [
            parse_ego_vehicle_state(
                scenario.get_ego_state_at_iteration(i), nuplan_center
            )
            for i in range(start_iter, stop_iter)
        ]
    for i in range(len(data) - 1):
        data[i]["angular_velocity"] = compute_angular_velocity(
            initial_heading=data[i]["heading"],
            final_heading=data[i + 1]["heading"],
            dt=scenario.database_interval,
        )
    return data


def get_route_lane_polylines_from_roadblock_ids(
    map_api: AbstractMap, point: Point2D, radius: float, route_roadblock_ids: List[str]
) -> Union[List[List[List[float]]], List[int]]:
    """
    Mostly copied from: nuplan.planning.training.preprocessing.feature_builders.vector_builder_utils

    Extract route polylines from map for route specified by list of roadblock ids. Route is represented as collection of
        baseline polylines of all children lane/lane connectors or roadblock/roadblock connectors encompassing route.
    :param map_api: map to perform extraction on.
    :param point: [m] x, y coordinates in global frame.
    :param radius: [m] floating number about extraction query range.
    :param route_roadblock_ids: ids of roadblocks/roadblock connectors specifying route.
    :return: A route as sequence of lane/lane connector polylines AND lane/lane connector ids.
    """
    # shape: [num_lanes, num_points_per_lane (variable), 2]
    route_lane_polylines: List[List[List[float]]] = []
    map_objects = []
    map_objects_id = []

    # extract roadblocks/connectors within query radius to limit route consideration
    layer_names = [SemanticMapLayer.ROADBLOCK, SemanticMapLayer.ROADBLOCK_CONNECTOR]
    layers = map_api.get_proximal_map_objects(point, radius, layer_names)
    roadblock_ids: Set[str] = set()

    for layer_name in layer_names:
        roadblock_ids = roadblock_ids.union(
            {map_object.id for map_object in layers[layer_name]}
        )
    # prune route by connected roadblocks within query radius
    route_roadblock_ids = prune_route_by_connectivity(
        route_roadblock_ids, roadblock_ids
    )

    for route_roadblock_id in route_roadblock_ids:
        # roadblock
        roadblock_obj = map_api.get_map_object(
            route_roadblock_id, SemanticMapLayer.ROADBLOCK
        )

        # roadblock connector
        if not roadblock_obj:
            roadblock_obj = map_api.get_map_object(
                route_roadblock_id, SemanticMapLayer.ROADBLOCK_CONNECTOR
            )

        # represent roadblock/connector by interior lanes/connectors
        if roadblock_obj:
            map_objects += roadblock_obj.interior_edges

    # sort by distance to query point
    map_objects.sort(
        key=lambda map_obj: float(
            get_distance_between_map_object_and_point(point, map_obj)
        )
    )

    for map_obj in map_objects:
        # print(f"centerline length: {map_obj.baseline_path.length}")
        # print(f"centerline points: {len(map_obj.baseline_path.discrete_path)}")
        map_objects_id.append(int(map_obj.id))
        baseline_path_polyline = [
            [node.x, node.y] for node in map_obj.baseline_path.discrete_path
        ]
        route_lane_polylines.append(baseline_path_polyline)

    return route_lane_polylines, map_objects_id


def get_scenario_start_iter_tuple(scenario, n_step):
    """
    Sample subscenarios from the scenario based on the desired new length (n_step).
    Start at the beginning (t=0) and go to the end in 1 sec steps until
    the desired length reaches the end of the scenario.
    """
    scenario_id_start_idx_tuples = []
    scenario_len_sec = int(scenario.duration_s.time_s)
    scenario_time_step = scenario.database_interval
    assert scenario_time_step == 0.1, "Only support 0.1s time step"
    last_iteration = int((scenario_len_sec - 1) / scenario_time_step) - (n_step - 1)
    for iteration in range(0, last_iteration, 10):
        scenario_id_start_idx_tuples.append((scenario, iteration))
    return scenario_id_start_idx_tuples


def fill_track_with_state(
    track_state: dict, state: dict, current_timestep: int
) -> dict:
    """
    Fills a track with the information from the track_obj
    :param track: the track to fill
    :param track_obj: the track object
    :return: the filled track
    """
    track_state["position_x"][current_timestep] = state["position"][0]
    track_state["position_y"][current_timestep] = state["position"][1]
    track_state["heading"][current_timestep] = state["heading"]
    track_state["velocity_x"][current_timestep] = state["velocity"][0]
    track_state["velocity_y"][current_timestep] = state["velocity"][1]
    track_state["valid"][current_timestep] = state["valid"]
    track_state["length"][current_timestep] = state["length"]
    track_state["width"][current_timestep] = state["width"]
    track_state["height"][current_timestep] = state["height"]


def calc_velocity_from_positions(track_state: dict, dt: float) -> None:
    positions = np.hstack([track_state["position_x"], track_state["position_y"]])
    velocity = (positions[1:] - positions[:-1]) / dt
    track_state["velocity_x"][:-1] = velocity[..., 0]
    track_state["velocity_x"][-1] = track_state["velocity_x"][-2]
    track_state["velocity_y"][:-1] = velocity[..., 1]
    track_state["velocity_y"][-1] = track_state["velocity_y"][-2]


def mining_for_interesting_agents(
    tracks: dict,
    n_agent_prediction_challenge: int,
    n_agent_interaction_challenge: int,
) -> dict:
    """
    Here, we will mine for interesting agents to be used in the prediction and interaction challenges.

    Principal from Waymo Open Motion Dataset (https://arxiv.org/pdf/2104.10133): "The training, validation,
    and testing sets contain up to 8 objects per scenario chosen to include at least 2 objects of each
    type if available. Selection is biased to include objects that do not follow a constant velocity model
    or straight paths. For the validation interactive and testing interactive sets, only the mined
    interactive agent pair objects are included in the list of objects to predict."

    The above principle is not strictly implemented, but the idea is tried to be followed.
    """
    prediction_agents_bic = FixedLengthDict(n_agent_prediction_challenge - 1)
    prediction_agents_ped = FixedLengthDict(n_agent_prediction_challenge - 1)
    prediction_agents_veh = FixedLengthDict(n_agent_prediction_challenge - 1)

    interact_agents_bic = FixedLengthDict(n_agent_interaction_challenge - 1)
    interact_agents_ped = FixedLengthDict(n_agent_interaction_challenge - 1)
    interact_agents_veh = FixedLengthDict(n_agent_interaction_challenge - 1)

    for id, track in tracks.items():
        type = track["type"]
        traj_score = calculate_interest_score(track)
        # search for vehicles
        if type == 0:
            if id == "ego":
                pass
            else:
                prediction_agents_veh.add(id, traj_score)
                interact_agents_veh.add(id, traj_score)
        # search for pedestrians
        elif type == 1:
            prediction_agents_ped.add(id, traj_score)
            interact_agents_ped.add(id, traj_score)
        # search for ciclists
        elif type == 2:
            prediction_agents_bic.add(id, traj_score)
            interact_agents_bic.add(id, traj_score)

    # Create list with ids of agents for prediction challenge
    track_ids_predict = ["ego"]
    track_ids_predict.extend(prediction_agents_bic.get_top_x_keys(2))
    track_ids_predict.extend(prediction_agents_ped.get_top_x_keys(2))
    len_track_ids_predict = len(track_ids_predict)
    track_ids_predict.extend(
        prediction_agents_veh.get_top_x_keys(
            n_agent_prediction_challenge - len_track_ids_predict
        )
    )
    # fill with pedestrians (first) and biclists (second) if necessary
    if len(track_ids_predict) < n_agent_prediction_challenge:
        track_ids_predict.extend(
            prediction_agents_ped.get_top_x_keys(
                n_agent_prediction_challenge - len(track_ids_predict)
            )
        )
    if len(track_ids_predict) < n_agent_prediction_challenge:
        track_ids_predict.extend(
            prediction_agents_bic.get_top_x_keys(
                n_agent_prediction_challenge - len(track_ids_predict)
            )
        )

    # TODO: Get the most interactive agent for the ego vehicle
    # Currently only the most interesting agetent is selected
    track_ids_interact = ["ego"]
    track_ids_interact.extend(
        interact_agents_veh.get_top_x_keys(n_agent_interaction_challenge - 1)
    )
    # fill with pedestrians (first) and biclists (second) if necessary
    if len(track_ids_interact) < n_agent_interaction_challenge:
        track_ids_interact.extend(
            interact_agents_ped.get_top_x_keys(
                n_agent_interaction_challenge - len(track_ids_interact)
            )
        )
    if len(track_ids_interact) < n_agent_interaction_challenge:
        track_ids_interact.extend(
            interact_agents_bic.get_top_x_keys(
                n_agent_interaction_challenge - len(track_ids_interact)
            )
        )

    return track_ids_predict, track_ids_interact


def calculate_interest_score(track: dict) -> float:
    """
    Calculate the interest score for a trajectory based on the following metrics:
        - Total heading change
        - Lateral deviation
        - Total acceleration
        - Progress
    """
    valid = track["state"]["valid"]
    position_x = track["state"]["position_x"][valid]
    position_y = track["state"]["position_y"][valid]
    heading = track["state"]["heading"][valid]

    # 1. Compute Euclidean distance traveled
    dx = np.diff(position_x)
    dy = np.diff(position_y)
    distances = np.sqrt(dx**2 + dy**2)
    total_progress = np.sum(distances)

    # 2. Compute heading changes (angle of trajectory)
    angles = np.arctan2(dy, dx)  # Heading angle at each timestep
    heading_changes = np.abs(np.diff(angles))  # Absolute change in heading
    total_heading_change = np.sum(heading_changes)

    # 3. Compute lateral deviations for lane changes (approximation)
    lateral_deviation = normal_distance_2d_with_angle(
        [position_x[-1], position_y[-1]], [position_x[0], position_y[0]], heading[0]
    )  # Standard deviation of lateral movement

    # 4. Compute acceleration changes
    velocity_x = np.diff(position_x)  # First derivative (velocity)
    velocity_y = np.diff(position_y)
    acceleration_x = np.diff(velocity_x)  # Second derivative (acceleration)
    acceleration_y = np.diff(velocity_y)
    total_acceleration = np.sum(np.sqrt(acceleration_x**2 + acceleration_y**2))

    # Combine scores with weights
    interest_score = (
        0.5 * total_heading_change  # Turning
        + 1 * lateral_deviation  # Lane changes
        + 2 * total_acceleration  # Acceleration/Deceleration
        + 0.1 * total_progress  # Progress
    )

    return interest_score


class FixedLengthDict:
    def __init__(self, max_length):
        self.data = {}
        self.max_length = max_length

    def add(self, id, value):
        # Add the new id-value pair to the dictionary
        self.data[id] = value
        # If max length is exceeded, remove the smallest value (based on value)
        if len(self.data) > self.max_length:
            # Find the ID of the smallest value
            smallest_id = min(self.data, key=self.data.get)
            del self.data[smallest_id]

    def get_top_x_keys(self, x):
        # Get the keys of the x highest values
        if x <= 0:
            return []
        # Sort by values in descending order and get the top x keys
        sorted_items = sorted(self.data.items(), key=lambda item: item[1], reverse=True)
        top_x_keys = [key for key, value in sorted_items[:x]]
        for key in top_x_keys:
            del self.data[key]
        return top_x_keys

    def __repr__(self):
        return repr(self.data)


def normal_distance_2d_with_angle(point, line_point, angle):
    # Convert inputs to numpy arrays
    P = np.array(point)
    A = np.array(line_point)
    # Direction vector from angle
    v = np.array([np.cos(angle), np.sin(angle)])
    # Vector from A to P
    AP = P - A
    # Compute the numerator (perpendicular projection magnitude)
    numerator = abs(v[1] * AP[0] - v[0] * AP[1])
    # Compute the denominator (magnitude of the direction vector)
    denominator = np.linalg.norm(v)
    # Compute the distance
    distance = numerator / denominator
    return distance


def create_rectangle_from_points(points):
    min_x = points[:, 0].min()  # Smallest x
    max_x = points[:, 0].max()  # Largest x
    min_y = points[:, 1].min()  # Smallest y
    max_y = points[:, 1].max()  # Largest y

    return {"min_x": min_x, "max_x": max_x, "min_y": min_y, "max_y": max_y}


def is_point_in_rectangle(point, bounds):
    x, y = point
    return (
        bounds["min_x"] <= x <= bounds["max_x"]
        and bounds["min_y"] <= y <= bounds["max_y"]
    )


# only for example using
example_scenario_types = "[behind_pedestrian_on_pickup_dropoff,  \
                        near_multiple_vehicles, \
                        high_magnitude_jerk, \
                        crossed_by_vehicle, \
                        following_lane_with_lead, \
                        changing_lane_to_left, \
                        accelerating_at_traffic_light_without_lead, \
                        stopping_at_stop_sign_with_lead, \
                        traversing_narrow_lane, \
                        waiting_for_pedestrian_to_cross, \
                        starting_left_turn, \
                        starting_high_speed_turn, \
                        starting_unprotected_cross_turn, \
                        starting_protected_noncross_turn, \
                        on_pickup_dropoff]"

#   - accelerating_at_crosswalk
#   - accelerating_at_stop_sign
#   - accelerating_at_stop_sign_no_crosswalk
#   - accelerating_at_traffic_light
#   - accelerating_at_traffic_light_with_lead
#   - accelerating_at_traffic_light_without_lead
#   - behind_bike
#   - behind_long_vehicle
#   - behind_pedestrian_on_driveable
#   - behind_pedestrian_on_pickup_dropoff
#   - changing_lane
#   - changing_lane_to_left
#   - changing_lane_to_right
#   - changing_lane_with_lead
#   - changing_lane_with_trail
#   - crossed_by_bike
#   - crossed_by_vehicle
#   - following_lane_with_lead
#   - following_lane_with_slow_lead
#   - following_lane_without_lead
#   - high_lateral_acceleration
#   - high_magnitude_jerk
#   - high_magnitude_speed
#   - low_magnitude_speed
#   - medium_magnitude_speed
#   - near_barrier_on_driveable
#   - near_construction_zone_sign
#   - near_high_speed_vehicle
#   - near_long_vehicle
#   - near_multiple_bikes
#   - near_multiple_pedestrians
#   - near_multiple_vehicles
#   - near_pedestrian_at_pickup_dropoff
#   - near_pedestrian_on_crosswalk
#   - near_pedestrian_on_crosswalk_with_ego
#   - near_trafficcone_on_driveable
#   - on_all_way_stop_intersection
#   - on_carpark
#   - on_intersection
#   - on_pickup_dropoff
#   - on_stopline_crosswalk
#   - on_stopline_stop_sign
#   - on_stopline_traffic_light
#   - on_traffic_light_intersection
#   - starting_high_speed_turn
#   - starting_left_turn
#   - starting_low_speed_turn
#   - starting_protected_cross_turn
#   - starting_protected_noncross_turn
#   - starting_right_turn
#   - starting_straight_stop_sign_intersection_traversal
#   - starting_straight_traffic_light_intersection_traversal
#   - starting_u_turn
#   - starting_unprotected_cross_turn
#   - starting_unprotected_noncross_turn
#   - stationary
#   - stationary_at_crosswalk
#   - stationary_at_traffic_light_with_lead
#   - stationary_at_traffic_light_without_lead
#   - stationary_in_traffic
#   - stopping_at_crosswalk
#   - stopping_at_stop_sign_no_crosswalk
#   - stopping_at_stop_sign_with_lead
#   - stopping_at_stop_sign_without_lead
#   - stopping_at_traffic_light_with_lead
#   - stopping_at_traffic_light_without_lead
#   - stopping_with_lead
#   - traversing_crosswalk
#   - traversing_intersection
#   - traversing_narrow_lane
#   - traversing_pickup_dropoff
#   - traversing_traffic_light_intersection
#   - waiting_for_pedestrian_to_cross

all_scenario_types = (
    "[near_pedestrian_on_crosswalk_with_ego,"
    "near_trafficcone_on_driveable,  "
    "following_lane_with_lead, "
    "following_lane_with_slow_lead,  "
    "following_lane_without_lead]"
)
