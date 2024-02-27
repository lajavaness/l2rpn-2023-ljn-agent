import json
import logging
import os
import re
import sys
from pathlib import Path

import grid2op
import numpy as np
from grid2op.Chronics import MultifolderWithCache
from grid2op.Converter import Converter, ToVect
from grid2op.gym_compat import BoxGymObsSpace, GymEnv
from gymnasium.spaces import Discrete
from tqdm import tqdm

from .env_components.custom_spaces import GlobalTopoActionSpace

SEED_VALUE = 42
DATA_DIR = Path(Path(__file__).parents[1], "data/")
CHALLENGE_ENV = "l2rpn_idf_2023"
TRAIN_ENV_NAME = "l2rpn_idf_2023_train"
VAL_ENV_NAME = "l2rpn_idf_2023_val"

DEFAULT_OBS_ATTR_TO_KEEP = [
    "day_of_week",
    "hour_of_day",
    "minute_of_hour",
    "prod_p",
    "prod_v",
    "load_p",
    "load_q",
    "actual_dispatch",
    "target_dispatch",
    "topo_vect",
    "time_before_cooldown_line",
    "time_before_cooldown_sub",
    "rho",
    "timestep_overflow",
    "line_status",
    "storage_power",
    "storage_charge",
]

ACT_SPACE_MAP = {
    0: "action_12_unsafe",
    1: "action_N1_safe",
    2: "action_N1_interm",
    3: "action_N1_unsafe",
}


def read_seed_file(seed_file):
    seeds = []
    with open(seed_file) as f:
        for line in f.readlines():
            seeds.append(json.loads(line))
    return seeds


def configure_logging():
    """Configure the logging. The logs would be written to stdout, log.log and debug.log"""
    DEBUG = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <level>{message}</level>"  # noqa: E501
    INFO = "<level>{message}</level>"

    handlers = [
        {"sink": sys.stderr, "level": "INFO", "format": INFO},
        {"sink": "log.log", "level": "INFO", "format": DEBUG},
        {"sink": "debug.log", "level": "DEBUG", "format": DEBUG},
    ]
    if "pytest" in sys.modules:
        # Only activate stderr in unittest
        handlers = handlers[:1]

    logging.configure(handlers=handlers)

    # Intercept standard logging messages toward your Loguru sinks
    # https://loguru.readthedocs.io/en/stable/overview.html#entirely-compatible-with-standard-logging
    logging.basicConfig(handlers=[InterceptHandler()], level=logging.INFO)


class InterceptHandler(logging.Handler):
    def emit(self, record):
        # Ignore some over-verbose useless logs
        name = record.name.split(".")[0]
        if name in ("tensorboard"):
            return

        # Get corresponding Loguru level if it exists
        try:
            level = logging.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logging.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


ASSET_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets")


def load_topo_action_dict(file, env, by_area=True):
    if file.endswith(".npy"):
        vect_action_space = np.load(file)
    elif file.endswith(".npz"):
        vect_action_space = np.load(file)["action_space"]
    else:
        raise ("Wrong file extension : provide .npy or .npz file with the right format")

    converter = ToVect(env.action_space)
    best_actions_list = []
    for i in tqdm(range(len(vect_action_space))):
        best_actions_list.append(converter.convert_act(vect_action_space[i]))

    if by_area:
        sub_by_area = env._game_rules.legal_action.substations_id_by_area
        action_by_area = {i: [] for i in range(len(sub_by_area.keys()))}

        for i, act in enumerate(best_actions_list):
            sub_id = int(act.as_dict()["set_bus_vect"]["modif_subs_id"][0])
            for i, subs in sub_by_area.items():
                if sub_id in subs:
                    action_by_area[i].append(act)

        return action_by_area

    else:
        return {"all_best_actions": best_actions_list}


def run_episode(env, agent):
    obs = env.reset()
    reward = env.reward_range[0]
    done = False
    step_count = 0
    while not done:
        # here you loop on the time steps: at each step your agent receive an observation
        # takes an action
        # and the environment computes the next observation that will be used at the next step.
        act = agent.act(obs, reward, done)
        print(act)
        obs, reward, done, info = env.step(act)
        print(info)
        step_count += 1
    print(step_count)
    return obs, info


class TopoMultiActConverter(Converter):
    def __init__(self, action_space, action_dict_by_area):
        super().__init__(action_space)
        self.action_space = action_space
        self.action_dict_by_area = action_dict_by_area
        self.id_map = [f"agent_{i}" for i in action_dict_by_area.keys()]

    def convert_act(self, encoded_act):
        regular_act = self.action_space({})
        for i, id_act in enumerate(encoded_act):
            regular_act += self.action_dict_by_area[i][id_act]
        return regular_act


def build_gym_env(env_name, action_space_path, **env_kwargs):
    """
    Create standard gym env with global topo action space
    """

    env = grid2op.make(env_name, **env_kwargs)
    if (
        "chronics_class" in env_kwargs
        and env_kwargs["chronics_class"] == MultifolderWithCache
    ):
        env.chronics_handler.real_data.set_filter(
            lambda x: re.match(".*_0$", x) is not None
        )
        env.chronics_handler.real_data.reset()
    env_gym = GymEnv(env)
    env_gym.observation_space.close()
    env_gym.observation_space = BoxGymObsSpace(
        env.observation_space, attr_to_keep=DEFAULT_OBS_ATTR_TO_KEEP
    )
    env_gym.action_space.close()
    topo_actions_dict = load_topo_action_dict(action_space_path, env, by_area=False)[
        "all_best_actions"
    ]
    print(len(topo_actions_dict))
    env_gym.action_space = GlobalTopoActionSpace(topo_actions_dict, env.action_space)
    action_space = Discrete(len(topo_actions_dict))

    return env_gym, action_space


def reduce_obs_vect(x, env, gym_obs_space):
    """
    Reduce the size of obs gym vector to the target size specified in env_gym
    """
    for i, obs in enumerate(x["obs"]):
        obs_g2op = env.observation_space.from_vect(np.array(obs))
        x["obs"][i] = gym_obs_space.to_gym(obs_g2op)
    return x
