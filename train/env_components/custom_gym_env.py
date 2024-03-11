# Copyright (c) 2023-2024 La Javaness (https://lajavaness.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN 2023 LJN Agent, a repository for the winning agent of L2RPN 2023 competition. It is a submodule contribution to the L2RPN Baselines repository.

from typing import List

import numpy as np
from grid2op.Action import BaseAction
from l2rpn_baselines.utils import GymEnvWithHeuristics


class GymEnvForN1training(GymEnvWithHeuristics):
    def __init__(
        self,
        env_init,
        optim_agent,
        *args,
        reward_cumul="init",
        safe_max_rho=0.9,
        danger_rho=0.99,
        **kwargs,
    ):
        super().__init__(env_init, reward_cumul=reward_cumul, *args, **kwargs)
        self._safe_max_rho = safe_max_rho
        self._danger_rho = danger_rho
        self.optim_agent = optim_agent

    def heuristic_actions(self, g2op_obs, reward, done, info) -> List[BaseAction]:
        to_reco = (g2op_obs.time_before_cooldown_line == 0) & (~g2op_obs.line_status)
        res = []
        if np.any(to_reco):
            # reconnect something if it can be
            reco_id = np.where(to_reco)[0]
            for line_id in reco_id:
                g2op_act = self.init_env.action_space(
                    {"set_line_status": [(line_id, +1)]}
                )
                res.append(g2op_act)
        elif g2op_obs.rho.max() <= self._safe_max_rho:
            # play do nothing if there is "no problem" according to the "rule of thumb"
            res = [self.init_env.action_space()]
        elif g2op_obs.rho.max() >= self._danger_rho:
            g2op_act = self.optim_agent.act(g2op_obs, reward, done)
            res = [g2op_act]

        return res
