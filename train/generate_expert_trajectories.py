# Copyright (c) 2023-2024 La Javaness (https://lajavaness.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN 2023 LJN Agent, a repository for the winning agent of L2RPN 2023 competition. It is a submodule contribution to the L2RPN Baselines repository.

import json
import os
from functools import partial

import grid2op
import numpy as np
from datasets import Dataset
from grid2op.gym_compat import GymEnv

from .utils import ACT_SPACE_MAP, DEFAULT_OBS_ATTR_TO_KEEP, reduce_obs_vect


class ExpertDatasetBuilder:
    """
    This class handles the generation of an expert trajectories dataset for a given action spacebased on the logs
    data generated with the grid2op runner.
    When instantiating the class, one must specify an action space and an gym ideally implementing proper heuristics.
    """

    def __init__(self, log_path: str, env_gym: GymEnv, act_space_path: str):
        self.log_path = log_path
        self.env_gym = env_gym
        self.act_space_vect = np.load(act_space_path)["action_space"]

    def _get_hist_episode(
        self,
        episode: str,
        keyword: str,
    ):
        """Loads npz file from the log directory of an episode based on the keyword

        Parameters
        ----------
        episode : str
            Chronic's name
        keyword : str
            Keyword should be either : observations, actions, rewards.

        Returns
        -------
        np.array
            Array of vectors containing historic for the specifed episode.
        """
        path = os.path.join(self.log_path, episode, keyword + ".npz")
        assert os.path.exists(path)
        return np.load(path)["data"]

    def vect_to_gym_act_episode(self, act_hist: np.array):
        """
        Generate a list of discrete gym actions from grid2op vector actions data
        by finding matching vector actions within the provided action space.

        Parameters
        ----------
        act_hist : np.array
            Array representing the actions from a given episode (usually stored as actions.npz by the runner)
        Returns
        -------
        list
            List of gym actions within the specified action space.
            If the action is not in the action space the default index is -1.
        """
        matching_indices = []
        for row_a in act_hist:
            if np.any(np.isnan(row_a)):
                return matching_indices
            match_index = np.where(np.all(self.act_space_vect == row_a, axis=1))[0]
            if len(match_index) > 0:
                matching_indices.append(match_index[0])
            else:
                matching_indices.append(-1)
        return matching_indices

    def _get_episode_data(self, episode: str):
        """
        Returns obs, acts and rews array for a given episode and action space.

        Parameters
        ----------
        episode : str
            Episode name as written in the dataset / runner logs

        Returns
        -------
        dict
            Episode data containing actions, observations, rewards, terminal state and info (required for imitation learning)
        """
        episode_data = {}

        act = self._get_hist_episode(episode, "actions")
        obs = self._get_hist_episode(episode, "observations")
        rews = self._get_hist_episode(episode, "rewards")

        with open(os.path.join(self.log_path, episode, "episode_meta.json"), "r") as f:
            metadata = json.load(f)
        time_step_played = metadata["nb_timestep_played"]

        episode_data["acts"] = self.vect_to_gym_act_episode(act)[:time_step_played]
        filter_acts = np.array(episode_data["acts"]) != -1

        episode_data["acts"] = np.array(episode_data["acts"])[filter_acts]
        episode_data["rews"] = np.array(rews[:time_step_played])[filter_acts]

        # Add observation index after last relevant topo action
        if True in filter_acts:
            last_true_index = np.where(filter_acts)[0][-1]
            filter_acts = np.concatenate((filter_acts, np.array([False])))
            filter_acts[last_true_index + 1] = True
        else:
            filter_acts = np.concatenate((np.array([True]), filter_acts))
        episode_data["obs"] = np.array(obs[: time_step_played + 1])[filter_acts]
        episode_data["terminal"] = True
        episode_data["infos"] = ["{}" for i in range(len(episode_data["rews"]))]

        return episode_data

    def data_generator(self):
        """Generator used by the hugging face's datasets package to generate the trajectories dataset

        Yields
        ------
        dict
            Episode data
        """
        episode_list = [
            ep
            for ep in os.listdir(self.log_path)
            if os.path.isdir(os.path.join(self.log_path, ep))
        ]
        for ep in episode_list:
            yield self._get_episode_data(ep)


if __name__ == "__main__":
    from grid2op.gym_compat import BoxGymObsSpace
    from lightsim2grid.lightSimBackend import LightSimBackend
    from tqdm import tqdm

    # Path variables
    AGENT_LOG_PATH = os.path.join("./agents_log_BC")
    DATASET_PATH = os.path.join("./expert_datasets")
    if not os.path.exists(DATASET_PATH):
        os.mkdir(DATASET_PATH)

    # Env creation
    ACTION_SPACE_PATH = "./l2rpn-2023-ljn-agent/action"
    env = grid2op.make("l2rpn_idf_2023_train", backend=LightSimBackend())
    env_gym = GymEnv(env)
    env_gym.observation_space.close()
    env_gym.observation_space = BoxGymObsSpace(
        env.observation_space, attr_to_keep=DEFAULT_OBS_ATTR_TO_KEEP
    )

    reduc_vec_func = partial(
        reduce_obs_vect, env=env, gym_obs_space=env_gym.observation_space
    )

    ### Generate imitation dataset for the 4 different action spaces ###
    for act_space in tqdm(ACT_SPACE_MAP.values()):
        data_builder = ExpertDatasetBuilder(
            AGENT_LOG_PATH,
            env_gym,
            act_space_path=os.path.join(ACTION_SPACE_PATH, f"{act_space}.npz"),
        )
        gen = data_builder.data_generator
        data = Dataset.from_generator(gen)

        # Post-processing
        data = data.filter(lambda x: len(x["acts"]) > 0)
        data = data.map(reduc_vec_func)

        data.save_to_disk(os.path.join(DATASET_PATH, act_space))
