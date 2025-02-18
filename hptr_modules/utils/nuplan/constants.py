TL_TYPES = {
    "GREEN": 3,
    "YELLOW": 2,
    "RED": 1,
    "UNKNOWN": 0,
    # "FLASHING": 4,
}
N_TL_STATE = len(TL_TYPES)

AGENT_TYPES = {
    "VEHICLE": 0,  # Includes all four or more wheeled vehicles, as well as trailers.
    "PEDESTRIAN": 1,  # All types of pedestrians, incl. strollers and wheelchairs.
    "BICYCLE": 2,  # Includes bicycles, motorcycles and tricycles.
    "TRAFFIC_CONE": 3,  # Cones that are temporarily placed to control the flow of traffic.
    "BARRIER": 3,  # Solid barriers that can be either temporary or permanent.
    "CZONE_SIGN": 3,  # Temporary signs that indicate construction zones.
    "GENERIC_OBJECT": 3,  # Animals, debris, pushable/pullable objects, permanent poles.
    "EGO": 0,  # The ego vehicle.
}
N_AGENT_TYPE = len(set(AGENT_TYPES.values()))

N_PL_MAX = 2500
N_TL_MAX = 40
N_AGENT_MAX = 1000
N_PL_ROUTE_MAX = 250

N_PL = 1024
N_TL = 200  # due to polyline splitting this value can be higher than N_TL_MAX
N_AGENT = 64
N_AGENT_NO_SIM = N_AGENT_MAX - N_AGENT
N_PL_ROUTE = N_PL_ROUTE_MAX

THRESH_MAP = 120
THRESH_AGENT = 120

N_STEP = 91
STEP_CURRENT = 10

N_SDC_AGENT = 1
N_AGENT_PRED_CHALLENGE = 8
N_AGENT_INTERACT_CHALLENGE = 2

PL_TYPES = {
    "INTERSECTION": 0,
    "STOP_LINE": 1,
    "CROSSWALK": 2,
    "WALKWAYS": 3,
    "BOUNDARIES": 4,
    "CARPARK_AREA": 5,
    "CENTERLINE": 6,
    "ROUTE": 7,
}
N_PL_TYPE = len(PL_TYPES)
DIM_VEH_LANES = [6]
DIM_CYC_LANES = [4, 6]
DIM_PED_LANES = [0, 2, 3, 4]
