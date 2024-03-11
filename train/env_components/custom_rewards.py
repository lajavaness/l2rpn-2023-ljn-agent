# Copyright (c) 2023-2024 La Javaness (https://lajavaness.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN 2023 LJN Agent, a repository for the winning agent of L2RPN 2023 competition. It is a submodule contribution to the L2RPN Baselines repository.

import grid2op
from grid2op.Action import BaseAction
from grid2op.dtypes import dt_float
from grid2op.Environment import Environment
from grid2op.Reward import BaseReward


class PPO_Reward(BaseReward):
    def __init__(self):
        """
        PPO_Reward class, based on the BaseReward from Grid2Op
        """
        BaseReward.__init__(self)
        self.reward_min = -10
        self.reward_std = 2

    def __call__(
        self,
        action: BaseAction,
        env: Environment,
        has_error: bool,
        is_done: bool,
        is_illegal: bool,
        is_ambiguous: bool,
    ) -> float:
        if is_done or is_illegal or is_ambiguous or has_error:
            return self.reward_min
        rho_max = env.get_obs().rho.max()
        return self.reward_std - rho_max * (1 if rho_max < 0.95 else 2)
