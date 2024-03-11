# Copyright (c) 2023-2024 La Javaness (https://lajavaness.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN 2023 LJN Agent, a repository for the winning agent of L2RPN 2023 competition. It is a submodule contribution to the L2RPN Baselines repository.

from functools import reduce
from typing import List

import numpy as np
from grid2op.gym_compat import GymEnv
from gymnasium.spaces import Dict, Discrete, MultiBinary, MultiDiscrete

### -------------------------------------- ###
### This file provides some custom gym act ###
### space to be used for imitation and RL  ###
### based training                         ###
### -------------------------------------- ###


class MultiAgentTopoActionSpace(Dict):
    """
    Gym action space for multi-agent setup, inherit from Dict gymnasium space.
    ----> Currently not used.
    """

    def __init__(self, topo_actions_dict: dict, g2op_action_space):
        Dict.__init__(
            self,
            {
                f"agent_{i}": Discrete(len(topo_actions_dict[i]), start=-1)
                for i in topo_actions_dict.keys()
            },
        )
        self.topo_actions_dict = {
            f"agent_{i}": topo_actions_dict[i] for i in topo_actions_dict.keys()
        }
        self.g2op_action_space = g2op_action_space

    def from_gym(self, gym_action):
        act = self.g2op_action_space({})
        for agent, id in gym_action.items():
            if id != -1:
                act += self.topo_actions_dict[agent][id]
        return act

    def close(self):
        if hasattr(self, "_init_env"):
            self._init_env = None  # this doesn't own the environment


class MultiAreaTopoActionSpace(MultiDiscrete):
    """
    Encode a reduced topology action space as a multi-discrete gym space with a size corresponding to the number of sub-area.
    Hence a topology action in with this gym space is a tuple of index where each index corresponds to a bus reconfiguration in a specified sub-area.
    ---> Currently not used, but has been used successfully during the challenge, can be used with the l2rpn_idf_2023 env.
    """

    def __init__(self, topo_actions_dict: dict, g2op_action_space):
        MultiDiscrete.__init__(
            self, [len(topo_actions_dict[i]) + 1 for i in topo_actions_dict.keys()]
        )
        self.topo_actions_dict = topo_actions_dict
        self.g2op_action_space = g2op_action_space

    def from_gym(self, gym_action):
        act = self.g2op_action_space({})
        for i, id in enumerate(gym_action):
            # print(i,id)
            if 0 < id < len(self.topo_actions_dict[i]) + 1:
                act += self.topo_actions_dict[i][id - 1]
        return act

    def close(self):
        if hasattr(self, "_init_env"):
            self._init_env = None  # this doesn't own the environment


class GlobalTopoActionSpace(Discrete):
    """
    Simple action space encoding a reduced topo action space a Discrete gym space.
    An index corresponds to a single topological action.
    """

    def __init__(self, topo_actions_list: list, g2op_action_space):
        Discrete.__init__(self, len(topo_actions_list))
        self.topo_actions_list = topo_actions_list
        self.g2op_action_space = g2op_action_space

    def from_gym(self, gym_action):
        return self.topo_actions_list[int(gym_action)]

    def close(self):
        if hasattr(self, "_init_env"):
            self._init_env = None  # this doesn't own the environment


class MultiAlertActionSpace(MultiBinary):
    """
    MultiBinary space dedicated to Alert. Each variable represents an alertable lines.
    Should be used to train an Alert raiser policy.
    ---> currently not used
    """

    def __init__(self, g2op_action_space, attackable_lines):
        size = reduce(lambda count, l: count + len(l), attackable_lines, 0)
        MultiBinary.__init__(self, size)
        self.attackable_lines = attackable_lines
        self.g2op_action_space = g2op_action_space

    def from_gym(self, gym_action):
        id_to_raise = np.where(np.array(gym_action) == 1)[0]
        act = self.g2op_action_space({"raise_alert": list(id_to_raise)})
        return act

    def close(self):
        if hasattr(self, "_init_env"):
            self._init_env = None  # this doesn't own the environment
