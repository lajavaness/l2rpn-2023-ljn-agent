# Copyright (c) 2023-2024 La Javaness (https://lajavaness.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN 2023 LJN Agent, a repository for the winning agent of L2RPN 2023 competition. It is a submodule contribution to the L2RPN Baselines repository.

"""
Given an observation, this tutor does DCOPF optimization to get the best continuous actions.

This code is mainly based on l2rpn_baselines.OptimCVXPY, with the following differences:
- modify the workflow of the original code, mainly on act() method.
- include greedy search of discrete actions. 
- revise and comment the code to make it clearer.
"""

import copy
import logging
import os
import time
import warnings

import cvxpy as cp
import numpy as np
from grid2op.Action import BaseAction
from grid2op.Agent import BaseAgent
from grid2op.Backend import PandaPowerBackend
from grid2op.l2rpn_utils.idf_2023 import ObservationIDF2023
from grid2op.Observation import BaseObservation
from l2rpn_baselines.LJNAgent.utilities import find_best_line_to_reconnect, revert_topo
from lightsim2grid import LightSimBackend
from lightsim2grid.gridmodel import init

from .train.utils import build_gym_env

logger = logging.getLogger(__name__)

### ACTION SPACE PATH AND CONFIG USED DURING L2RPN 2023 IDF CHALLENGE ###

L2RPN_IDF_2023_DEFAULT_ACTION_SPACE = {
    "N1_safe": os.path.join(os.path.dirname(__file__), "action", "action_N1_safe.npz"),
    "N1_interm": os.path.join(
        os.path.dirname(__file__), "action", "action_N1_interm.npz"
    ),
    "N1_unsafe": os.path.join(
        os.path.dirname(__file__), "action", "action_N1_unsafe.npz"
    ),
    "12_unsafe": os.path.join(
        os.path.dirname(__file__), "action", "action_12_unsafe.npz"
    ),
}

L2RPN_IDF_2023_DEFAULT_OPTIM_CONFIG = {
    "margin_th_limit": 0.93,
    "alpha_por_error": 0.5,
    "rho_danger": 0.99,
    "rho_safe": 0.9,
    "penalty_curtailment_unsafe": 15,
    "penalty_redispatching_unsafe": 0.005,
    "penalty_storage_unsafe": 0.0075,
    "penalty_curtailment_safe": 0.0,
    "penalty_redispatching_safe": 0.0,
    "penalty_storage_safe": 0.0,
    "weight_redisp_target": 1.0,
    "weight_storage_target": 1.0,
    "weight_curtail_target": 1.0,
    "margin_rounding": 0.01,
    "margin_sparse": 5e-3,
    "max_iter": 100000,
    "areas": True,
    "sim_range_time_step": 1,
}


def convert_from_vect(action_space, act):
    """
    Helper to convert an action, represented as a numpy array as an :class:`grid2op.BaseAction` instance.

    Parameters
    ----------
    act: ``numppy.ndarray``
        An action cast as an :class:`grid2op.BaseAction.BaseAction` instance.

    Returns
    -------
    res: :class:`grid2op.Action.Action`
        The `act` parameters converted into a proper :class:`grid2op.BaseAction.BaseAction` object.
    """
    res = action_space({})
    res.from_vect(act)
    return res


def load_action_to_grid2op(
    action_space, action_vec_path, action_threshold=None, return_counts=False
):
    """Load actions saved in npz file and convert them to grid2op action class instances.

    Args:
        action_space: obtain from env.action_space
        action_vec_path: a path ending with .npz. The loaded all_actions has two keys 'action_space', 'counts'.
            all_actions['action_space'] is a (N, K) matrix, where each row is an action.
        action_threshold: if provided, will filter out actions with counts below this.
        return_counts: if True, return the counts.
    """
    # load all_actions from npz file with two keys 'action_space', 'counts'
    assert action_vec_path.endswith(".npz") and os.path.exists(
        action_vec_path
    ), FileNotFoundError(
        f"file_paths {action_vec_path} does not contain a valid npz file."
    )
    data = np.load(action_vec_path, allow_pickle=True)
    all_actions = data["action_space"]
    assert isinstance(all_actions, np.ndarray), RuntimeError(
        f"Expect {action_vec_path} to be an ndarray, got {type(all_actions)}"
    )
    action_dim = action_space.size()
    assert all_actions.shape[1] == action_dim, RuntimeError(
        f"Expect {action_vec_path} to be an ndarray of shape {action_dim}, got {all_actions.shape[1]}"
    )
    counts = data["counts"]
    if action_threshold is not None:
        num_actions = (counts >= action_threshold).sum()
        all_actions = all_actions[:num_actions]
        counts = counts[:num_actions]
        logger.info("{} actions loaded.".format(num_actions))
    all_actions = [convert_from_vect(action_space, action) for action in all_actions]
    if return_counts:
        return all_actions, counts
    return all_actions


class LJNAgent(BaseAgent):
    """
    This agent choses its action by resolving, at each `agent.act(...)` call an optimization routine
    that is then converted to a grid2op action.
    It has 3 main behaviours:
    - `safe grid`: when the grid is safe, it tries to reconnect any disconnected lines.
    - `unsafe grid`: when the grid is unsafe, it tries to set it back to a "safe" state (all flows
    below their thermal limit). The strategy is:
        - Try to reconnect lines
        - Try to reset the grid to reference state (all elements to bus one)
        - If back2ref does not work, seach topology action.
        - optimizing storage units, curtailment and redispatching.

    See OPTIMCVXPY_CONFIG for a list of default configs.
    """

    SOLVER_TYPES = [cp.OSQP, cp.SCS, cp.SCIPY]
    # OSQP gives the best convergence compared to the other solvers

    def __init__(
        self,
        env,
        action_space,
        action_space_path_N1_safe: str = L2RPN_IDF_2023_DEFAULT_ACTION_SPACE["N1_safe"],
        action_space_path_N1_interm: str = L2RPN_IDF_2023_DEFAULT_ACTION_SPACE[
            "N1_interm"
        ],
        action_space_path_N1_unsafe: str = L2RPN_IDF_2023_DEFAULT_ACTION_SPACE[
            "N1_unsafe"
        ],
        action_space_path_12_unsafe: str = L2RPN_IDF_2023_DEFAULT_ACTION_SPACE[
            "12_unsafe"
        ],
        config: dict = L2RPN_IDF_2023_DEFAULT_OPTIM_CONFIG,
        verbose: bool = True,
        topo_model_path_N1_safe: str = None,
        topo_model_path_N1_unsafe: str = None,
        topo_model_path_N1_interm: str = None,
        topo_model_path_12_unsafe: str = None,
    ):
        """
        Initialize the agent with the given environment, action space, configuration, and other optional parameters.

        Parameters:
        - env: The environment object.
        - action_space: The action space object.
        - action_space_path_N1_safe: File path for safe actions using N-1 strategy when rho rho < rho_safe.
        - action_space_path_N1_interm: File path for N1 safe actions using N-1 strategy when  rho_safe < rho < rho_danger.
        - action_space_path_N1_unsafe: File path for N1 unsafe actions using N-1 strategy when rho > rho_danger.
        - action_space_path_12_unsafe: File path for unsafe actions from greedy search for cas of attacking Lines and the overflow and reducing the load of the network.
        - config: Configuration dictionary.
        - time_step: Time step for optimization, default is 1.
        - verbose: Verbosity level, default is 1.
        """

        BaseAgent.__init__(self, action_space)
        self.env = env
        self.do_nothing = action_space({})
        self.config = config
        self._get_grid_info(env)
        self._init_params(env)
        self._load_topo_actions(
            action_space_path_N1_safe,
            action_space_path_N1_interm,
            action_space_path_N1_unsafe,
            action_space_path_12_unsafe,
        )
        self.lines_in_area = [
            list_ids
            for list_ids in env._game_rules.legal_action.lines_id_by_area.values()
        ]
        self.max_iter = config["max_iter"]
        self.flow_computed = np.full(env.n_line, np.NaN, dtype=float)
        self.time_step = config["sim_range_time_step"]
        self.area = self.config["areas"]

        if not verbose:
            logging.disable(logging.CRITICAL)

        self.gym_env = {}
        self.model = {}

        if isinstance(topo_model_path_12_unsafe, str):
            self._load_topo_model(topo_model_path_12_unsafe, "12_unsafe")
        if isinstance(topo_model_path_N1_safe, str):
            self._load_topo_model(topo_model_path_N1_safe, "N1_safe")
        if isinstance(topo_model_path_N1_unsafe, str):
            self._load_topo_model(topo_model_path_N1_unsafe, "N1_unsafe")
        if isinstance(topo_model_path_N1_interm, str):
            self._load_topo_model(topo_model_path_N1_interm, "N1_interm")

    def _load_topo_model(self, model_path, name):
        # Lazy import
        from imitation.algorithms import bc
        from stable_baselines3 import PPO

        action_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), f"action/action_{name}.npz"
        )
        model_path = model_path

        logging.info(
            f"Loading neural network based topology model for {name} action space..."
        )
        self.gym_env[name] = build_gym_env("l2rpn_idf_2023", action_path)[0]
        custom_objects = {
            "observation_space": self.gym_env[name].observation_space,
            "action_space": self.gym_env[name].action_space,
        }
        self.model[name] = PPO.load(model_path, custom_objects=custom_objects)
        # self.model[name].set_env(self.gym_env[name])
        # self.model[name] = bc.reconstruct_policy(model_path,)
        logging.info(
            "Model successfully loaded, it will be used to predict topo actions when needed."
        )

    def _get_grid_info(self, env):
        """
        Fetch information about the grid from the environment.

        Parameters:
        - env: The environment object.
        """
        self.n_line = env.n_line
        self.n_sub = env.n_sub
        self.n_load = env.n_load
        self.n_gen = env.n_gen
        self.n_storage = env.n_storage
        self.line_or_to_subid = copy.deepcopy(env.line_or_to_subid)
        self.line_ex_to_subid = copy.deepcopy(env.line_ex_to_subid)
        self.load_to_subid = copy.deepcopy(env.load_to_subid)
        self.gen_to_subid = copy.deepcopy(env.gen_to_subid)
        self.storage_to_subid = copy.deepcopy(env.storage_to_subid)
        self.storage_Emax = copy.deepcopy(env.storage_Emax)

    def _init_params(self, env):
        """
        Initialize various optimization parameters and CVXPY parameters based on the environment and configuration.

        Parameters:
        - env: The environment object.
        """
        self.margin_rounding = float(self.config["margin_rounding"])
        self.margin_sparse = float(self.config["margin_sparse"])
        self.rho_danger = float(self.config["rho_danger"])
        self.rho_safe = float(self.config["rho_safe"])
        self._margin_th_limit = cp.Parameter(
            value=self.config["margin_th_limit"], nonneg=True
        )
        self._penalty_curtailment_unsafe = cp.Parameter(
            value=self.config["penalty_curtailment_unsafe"], nonneg=True
        )
        self._penalty_redispatching_unsafe = cp.Parameter(
            value=self.config["penalty_redispatching_unsafe"], nonneg=True
        )
        self._penalty_storage_unsafe = cp.Parameter(
            value=self.config["penalty_storage_unsafe"], nonneg=True
        )
        self._penalty_curtailment_safe = cp.Parameter(
            value=self.config["penalty_curtailment_safe"], nonneg=True
        )
        self._penalty_redispatching_safe = cp.Parameter(
            value=self.config["penalty_redispatching_safe"], nonneg=True
        )
        self._penalty_storage_safe = cp.Parameter(
            value=self.config["penalty_storage_safe"], nonneg=True
        )
        self._weight_redisp_target = cp.Parameter(
            value=self.config["weight_redisp_target"], nonneg=True
        )
        self._weight_storage_target = cp.Parameter(
            value=self.config["weight_storage_target"], nonneg=True
        )
        self._weight_curtail_target = cp.Parameter(
            value=self.config["weight_curtail_target"], nonneg=True
        )
        self._alpha_por_error = cp.Parameter(
            value=self.config["alpha_por_error"], nonneg=True
        )
        self.nb_max_bus = 2 * self.n_sub
        self._storage_setpoint = 0.5 * self.storage_Emax
        SoC = np.zeros(shape=self.nb_max_bus)
        for bus_id in range(self.nb_max_bus):
            SoC[bus_id] = (
                0.5 * self._storage_setpoint[self.storage_to_subid == bus_id].sum()
            )
        self._storage_target_bus = cp.Parameter(
            shape=self.nb_max_bus, value=1.0 * SoC, nonneg=True
        )
        self._storage_power_obs = cp.Parameter(value=0.0)
        powerlines_x, powerlines_g, powerlines_b, powerlines_ratio = (
            self._get_powerline_impedance(env)
        )
        self._powerlines_x = cp.Parameter(
            shape=powerlines_x.shape, value=1.0 * powerlines_x, pos=True
        )
        self._powerlines_g = cp.Parameter(
            shape=powerlines_x.shape, value=1.0 * powerlines_g, pos=True
        )
        self._powerlines_b = cp.Parameter(
            shape=powerlines_x.shape, value=1.0 * powerlines_b, neg=True
        )
        self._powerlines_ratio = cp.Parameter(
            shape=powerlines_x.shape, value=1.0 * powerlines_ratio, pos=True
        )
        self._prev_por_error = cp.Parameter(
            shape=powerlines_x.shape, value=np.zeros(env.n_line)
        )
        self.vm_or = cp.Parameter(
            shape=self.n_line, value=np.ones(self.n_line), pos=True
        )
        self.vm_ex = cp.Parameter(
            shape=self.n_line, value=np.ones(self.n_line), pos=True
        )
        self.bus_or = cp.Parameter(
            shape=self.n_line, value=1 * self.line_or_to_subid, integer=True
        )
        self.bus_ex = cp.Parameter(
            shape=self.n_line, value=1 * self.line_ex_to_subid, integer=True
        )
        self.bus_load = cp.Parameter(
            shape=self.n_load, value=1 * self.load_to_subid, integer=True
        )
        self.bus_gen = cp.Parameter(
            shape=self.n_gen, value=1 * self.gen_to_subid, integer=True
        )
        self.bus_storage = cp.Parameter(
            shape=self.n_storage, value=1 * self.storage_to_subid, integer=True
        )
        this_zeros_ = np.zeros(self.nb_max_bus)
        self.load_per_bus = cp.Parameter(
            shape=self.nb_max_bus, value=1.0 * this_zeros_, nonneg=True
        )
        self.gen_per_bus = cp.Parameter(
            shape=self.nb_max_bus, value=1.0 * this_zeros_, nonneg=True
        )
        self.redisp_up = cp.Parameter(
            shape=self.nb_max_bus, value=1.0 * this_zeros_, nonneg=True
        )
        self.redisp_down = cp.Parameter(
            shape=self.nb_max_bus, value=1.0 * this_zeros_, nonneg=True
        )
        self.curtail_down = cp.Parameter(
            shape=self.nb_max_bus, value=1.0 * this_zeros_, nonneg=True
        )
        self.curtail_up = cp.Parameter(
            shape=self.nb_max_bus, value=1.0 * this_zeros_, nonneg=True
        )
        self.storage_down = cp.Parameter(
            shape=self.nb_max_bus, value=1.0 * this_zeros_, nonneg=True
        )
        self.storage_up = cp.Parameter(
            shape=self.nb_max_bus, value=1.0 * this_zeros_, nonneg=True
        )
        self._th_lim_mw = cp.Parameter(
            shape=self.n_line, value=1.0 * env.get_thermal_limit(), nonneg=True
        )
        self._past_dispatch = cp.Parameter(
            shape=self.nb_max_bus, value=np.zeros(self.nb_max_bus)
        )
        self._past_state_of_charge = cp.Parameter(
            shape=self.nb_max_bus, value=np.zeros(self.nb_max_bus), nonneg=True
        )

    def _get_powerline_impedance(self, env):
        """
        Retrieve the impedance of power lines and transformers from the environment's backend.

        Parameters:
        - env: The environment object which contains the backend and grid information.

        Returns:
        - Tuple of numpy arrays containing reactance (x), conductance (g), susceptance (b), and powerline ratios.
        """
        if isinstance(env.backend, LightSimBackend):
            line_info = env.backend._grid.get_lines()
            trafo_info = env.backend._grid.get_trafos()
        elif isinstance(env.backend, PandaPowerBackend):
            pp_net = env.backend._grid
            grid_model = init(pp_net)
            line_info = grid_model.get_lines()
            trafo_info = grid_model.get_trafos()
        else:
            raise RuntimeError(
                f"Unkown backend type: {type(env.backend)}. If you want to use "
                "OptimCVXPY, you need to provide the reactance of each powerline / "
                "transformer in per unit in the `lines_x` parameter."
            )
        powerlines_x = np.array(
            [float(el.x_pu) for el in line_info] + [float(el.x_pu) for el in trafo_info]
        )
        powerlines_g = np.array(
            [(1 / (el.r_pu + 1j * el.x_pu)).real for el in line_info]
            + [(1 / (el.r_pu + 1j * el.x_pu)).real for el in trafo_info]
        )
        powerlines_b = np.array(
            [(1 / (el.r_pu + 1j * el.x_pu)).imag for el in line_info]
            + [(1 / (el.r_pu + 1j * el.x_pu)).imag for el in trafo_info]
        )
        powerlines_ratio = np.array(
            [1.0] * len(line_info) + [el.ratio for el in trafo_info]
        )
        return powerlines_x, powerlines_g, powerlines_b, powerlines_ratio

    def _load_topo_actions(
        self,
        action_space_path_N1_safe,
        action_space_path_N1_interm,
        action_space_path_N1_unsafe,
        action_space_path_12_unsafe,
    ):
        """
        Load predefined topology actions from the given paths and print the number of actions loaded for each category.

        Parameters:
        - action_space_path_N1_safe: File path for safe actions using N-1 strategy when rho rho < rho_safe.
        - action_space_path_N1_interm: File path for N1 safe actions using N-1 strategy when  rho_safe < rho < rho_danger.
        - action_space_path_N1_unsafe: File path for N1 unsafe actions using N-1 strategy when rho > rho_danger.
        - action_space_path_12_unsafe: File path for unsafe actions from greedy search for cas of attacking Lines and the overflow and reducing the load of the network.
        """
        self.topo_actions_to_check_N1_safe = load_action_to_grid2op(
            self.action_space, action_space_path_N1_safe
        )
        self.num_topo_actions_to_check_N1_safe = len(self.topo_actions_to_check_N1_safe)

        self.topo_actions_to_check_N1_interm = load_action_to_grid2op(
            self.action_space, action_space_path_N1_interm
        )
        self.num_topo_actions_to_check_N1_interm = len(
            self.topo_actions_to_check_N1_interm
        )

        self.topo_actions_to_check_N1_unsafe = load_action_to_grid2op(
            self.action_space, action_space_path_N1_unsafe
        )
        self.num_topo_actions_to_check_N1_unsafe = len(
            self.topo_actions_to_check_N1_unsafe
        )

        self.topo_actions_to_check_12_unsafe = load_action_to_grid2op(
            self.action_space, action_space_path_12_unsafe
        )
        self.num_topo_actions_to_check_12_unsafe = len(
            self.topo_actions_to_check_12_unsafe
        )

        logger.info("Load N1 safe:", self.num_topo_actions_to_check_N1_safe)
        logger.info("Load N1 interm:", self.num_topo_actions_to_check_N1_interm)
        logger.info("Load N1 unsafe:", self.num_topo_actions_to_check_N1_unsafe)
        logger.info("Load 12 interm:", self.num_topo_actions_to_check_12_unsafe)

    def _update_topo_param(self, obs: ObservationIDF2023):
        """
        Update the topology parameters based on the current observation.

        Parameters:
        - obs: The current observation of the system.
        """
        tmp_ = 1 * obs.line_or_to_subid
        tmp_[obs.line_or_bus == 2] += obs.n_sub
        self.bus_or.value[:] = tmp_
        tmp_ = 1 * obs.line_ex_to_subid
        tmp_[obs.line_ex_bus == 2] += obs.n_sub
        self.bus_ex.value[:] = tmp_

        # "disconnect" in the model the line disconnected
        # it should be equilavent to connect them all (at both side) to the slack
        self.bus_ex.value[(obs.line_or_bus == -1) | (obs.line_ex_bus == -1)] = 0
        self.bus_or.value[(obs.line_or_bus == -1) | (obs.line_ex_bus == -1)] = 0

        tmp_ = 1 * obs.load_to_subid
        tmp_[obs.load_bus == 2] += obs.n_sub
        self.bus_load.value[:] = tmp_

        tmp_ = 1 * obs.gen_to_subid
        tmp_[obs.gen_bus == 2] += obs.n_sub
        self.bus_gen.value[:] = tmp_

        if self.bus_storage is not None:
            tmp_ = 1 * obs.storage_to_subid
            tmp_[obs.storage_bus == 2] += obs.n_sub
            self.bus_storage.value[:] = tmp_

        # Normalize voltage according to standards
        self.vm_or.value[:] = np.array(
            [
                v_or / 138 if v_or < 147 else v_or / 161 if v_or < 171 else v_or / 345
                for v_or in obs.v_or
            ]
        )
        self.vm_ex.value[:] = np.array(
            [
                v_ex / 138 if v_ex < 147 else v_ex / 161 if v_ex < 171 else v_ex / 345
                for v_ex in obs.v_ex
            ]
        )

    # Updates the thermal limits of the powerlines.
    def _update_th_lim_param(self, obs: ObservationIDF2023):
        # Update the thermal limit
        threshold_ = 1.0
        self._th_lim_mw.value[:] = (
            0.001 * obs.thermal_limit
        ) ** 2 * obs.v_or**2 * 3.0 - obs.q_or**2
        mask_ok = self._th_lim_mw.value >= threshold_
        self._th_lim_mw.value[mask_ok] = np.sqrt(self._th_lim_mw.value[mask_ok])
        self._th_lim_mw.value[~mask_ok] = threshold_

    def _update_storage_power_obs(self, obs: ObservationIDF2023):
        """
        Update the storage power observation parameter.

        Parameters:
        - obs: The current observation of the system.
        """
        # self._storage_power_obs.value += obs.storage_power.sum()
        self._storage_power_obs.value = 0.0

    def _update_inj_param(self, obs: ObservationIDF2023):
        """
        Update the injection parameters for each bus based on the current observation.

        Parameters:
        - obs: The current observation of the system.
        """
        self.load_per_bus.value[:] = 0.0
        self.gen_per_bus.value[:] = 0.0
        load_p = 1.0 * obs.load_p
        load_p *= (obs.gen_p.sum() - self._storage_power_obs.value) / load_p.sum()
        for bus_id in range(self.nb_max_bus):
            self.load_per_bus.value[bus_id] += load_p[
                self.bus_load.value == bus_id
            ].sum()
            self.gen_per_bus.value[bus_id] += obs.gen_p[
                self.bus_gen.value == bus_id
            ].sum()

    def _add_redisp_const_per_bus(self, obs: ObservationIDF2023, bus_id: int):
        """
        Add redispatching constraints for a given bus.

        Parameters:
        - obs: The current observation of the system.
        - bus_id: Identifier of the bus.
        """
        self.redisp_up.value[bus_id] = obs.gen_margin_up[
            self.bus_gen.value == bus_id
        ].sum()
        self.redisp_down.value[bus_id] = obs.gen_margin_down[
            self.bus_gen.value == bus_id
        ].sum()

    # Adds storage constraints for each bus in the grid. This can involve setting boundaries or conditions for storage actions.
    def _add_storage_const_per_bus(self, obs: ObservationIDF2023, bus_id: int):
        """
        Add storage constraints for a given bus.

        Parameters:
        - obs: The current observation of the system.
        - bus_id: Identifier of the bus.
        """
        if self.bus_storage is None:
            return
        if obs.storage_max_p_prod is not None:
            stor_down = obs.storage_max_p_prod[self.bus_storage.value == bus_id].sum()
            stor_down = min(
                stor_down,
                obs.storage_charge[self.bus_storage.value == bus_id].sum()
                * (60.0 / obs.delta_time),
            )
            self.storage_down.value[bus_id] = stor_down
        else:
            self.storage_down.value[bus_id] = 0.0
        if obs.storage_max_p_absorb is not None:
            stor_up = obs.storage_max_p_absorb[self.bus_storage.value == bus_id].sum()
            stor_up = min(
                stor_up,
                (obs.storage_Emax - obs.storage_charge)[
                    self.bus_storage.value == bus_id
                ].sum()
                * (60.0 / obs.delta_time),
            )
            self.storage_up.value[bus_id] = stor_up
        else:
            self.storage_up.value[bus_id] = 0.0

    # Bring down all valid constraints by margin_rounding
    def _remove_margin_rounding(self):
        """
        Adjust several attributes by subtracting a predefined margin to ensure they do not get too close to their limits.
        """
        self.storage_down.value[
            self.storage_down.value > self.margin_rounding
        ] -= self.margin_rounding
        self.storage_up.value[
            self.storage_up.value > self.margin_rounding
        ] -= self.margin_rounding
        self.curtail_down.value[
            self.curtail_down.value > self.margin_rounding
        ] -= self.margin_rounding
        self.curtail_up.value[
            self.curtail_up.value > self.margin_rounding
        ] -= self.margin_rounding
        self.redisp_up.value[
            self.redisp_up.value > self.margin_rounding
        ] -= self.margin_rounding
        self.redisp_down.value[
            self.redisp_down.value > self.margin_rounding
        ] -= self.margin_rounding

    def _update_constraints_param_unsafe(self, obs: ObservationIDF2023):
        """
        Update constraints parameters for an "unsafe" state of the system.

        Parameters:
        - obs: The current observation of the system.
        """
        tmp_ = 1.0 * obs.gen_p
        tmp_[~obs.gen_renewable] = 0.0
        for bus_id in range(self.nb_max_bus):
            self._add_redisp_const_per_bus(obs, bus_id)
            mask_ = (self.bus_gen.value == bus_id) & obs.gen_renewable
            self.curtail_down.value[bus_id] = 0.0
            self.curtail_up.value[bus_id] = tmp_[mask_].sum()
            self._add_storage_const_per_bus(obs, bus_id)
        self._remove_margin_rounding()

    def _update_constraints_param_safe(self, obs):
        """
        Update constraints parameters for a "safe" state of the system.

        Parameters:
        - obs: The current observation of the system.
        """
        tmp_ = 1.0 * obs.gen_p
        tmp_[~obs.gen_renewable] = 0.0
        for bus_id in range(self.nb_max_bus):
            self._add_redisp_const_per_bus(obs, bus_id)
            mask_ = (self.bus_gen.value == bus_id) & obs.gen_renewable
            self.curtail_down.value[bus_id] = (
                obs.gen_p_before_curtail[mask_].sum() - tmp_[mask_].sum()
            )
            self._add_storage_const_per_bus(obs, bus_id)
            if self.bus_storage is not None:
                self._storage_target_bus.value[bus_id] = self._storage_setpoint[
                    self.bus_storage.value == bus_id
                ].sum()
            if self.bus_storage is not None:
                self._past_state_of_charge.value[bus_id] = obs.storage_charge[
                    self.bus_storage.value == bus_id
                ].sum()
            self._past_dispatch.value[bus_id] = obs.target_dispatch[
                self.bus_gen.value == bus_id
            ].sum()
        self.curtail_up.value[:] = 0.0  # never do more curtailment in "safe" mode
        self._remove_margin_rounding()

    def _validate_param_values(self):
        """
        Validate the values of different parameters to ensure they remain within valid limits.
        """
        self.storage_down._validate_value(self.storage_down.value)
        self.storage_up._validate_value(self.storage_up.value)
        self.curtail_down._validate_value(self.curtail_down.value)
        self.curtail_up._validate_value(self.curtail_up.value)
        self.redisp_up._validate_value(self.redisp_up.value)
        self.redisp_down._validate_value(self.redisp_down.value)
        self._th_lim_mw._validate_value(self._th_lim_mw.value)
        self._storage_target_bus._validate_value(self._storage_target_bus.value)
        self._past_dispatch._validate_value(self._past_dispatch.value)
        self._past_state_of_charge._validate_value(self._past_state_of_charge.value)

    def update_parameters(self, obs: ObservationIDF2023, safe: bool = False):
        """
        Update various parameters based on the current observation and the safety state of the system.

        Parameters:
        - obs: The current observation of the system.
        - safe: Boolean indicating if the system is in a "safe" state.
        """

        ## update the topology information
        self._update_topo_param(obs)

        ## update the thermal limit
        self._update_th_lim_param(obs)

        ## update the load / gen bus injected values
        self._update_inj_param(obs)

        ## update the constraints parameters
        if safe:
            self._update_constraints_param_safe(obs)
        else:
            self._update_constraints_param_unsafe(obs)

        ## check that all parameters have correct values
        ## for example non negative values for non negative parameters
        self._validate_param_values()

    def _aux_compute_kcl(self, inj_bus, f_or):
        """
        Compute Kirchhoff's Current Law (KCL) equations for every bus.

        Parameters:
        - inj_bus: Power injection at each bus.
        - f_or: Power flow on the origin side of each line.

        Returns:
        - List of KCL equations for each bus.
        """
        KCL_eq = []
        for bus_id in range(self.nb_max_bus):
            tmp = inj_bus[bus_id]
            if np.any(self.bus_or.value == bus_id):
                tmp += cp.sum(f_or[self.bus_or.value == bus_id])
            if np.any(self.bus_ex.value == bus_id):
                tmp -= cp.sum(f_or[self.bus_ex.value == bus_id])
            KCL_eq.append(tmp)
        return KCL_eq

    def _mask_theta_zero(self):
        """
        Find busbar that has no element connected to

        Returns:
        - A boolean array where `True` indicates the bus has a voltage angle of zero.
        """
        theta_is_zero = np.full(self.nb_max_bus, True, bool)
        theta_is_zero[self.bus_or.value] = False
        theta_is_zero[self.bus_ex.value] = False
        theta_is_zero[self.bus_load.value] = False
        theta_is_zero[self.bus_gen.value] = False
        if self.bus_storage is not None:
            theta_is_zero[self.bus_storage.value] = False
        theta_is_zero[0] = True  # slack bus
        return theta_is_zero

    def _solve_problem(self, prob, solver_type=None):
        """
        Attempt to solve the optimization problem using different solvers.

        Parameters:
        - prob: The optimization problem to be solved.
        - solver_type: The solver to be used (default: try all available solvers).

        Returns:
        - Boolean indicating if the solver has converged.
        """
        if solver_type is None:
            for solver_type in type(self).SOLVER_TYPES:
                res = self._solve_problem(prob, solver_type=solver_type)
                if res:
                    logger.info(
                        f"Solver {solver_type} has converged. Stopping solver search now."
                    )
                    return True
            return False
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                if solver_type is cp.OSQP:
                    tmp_ = prob.solve(
                        solver=solver_type,
                        verbose=0,
                        warm_start=False,
                        max_iter=self.max_iter,
                    )
                elif solver_type is cp.SCS:
                    tmp_ = prob.solve(
                        solver=solver_type, warm_start=False, max_iters=1000
                    )
                else:
                    tmp_ = prob.solve(solver=solver_type, warm_start=False)
            if np.isfinite(tmp_):
                return True
            else:
                logger.warning(
                    f"Problem diverged with dc approximation for {solver_type}, infinite value returned"
                )
                raise cp.error.SolverError("Infinite value")
        except cp.error.SolverError as exc_:
            logger.warning(
                f"Problem diverged with dc approximation for {solver_type}: {exc_}"
            )
            return False

    def run_dc(self, obs: ObservationIDF2023):
        """This method allows to perform a dc approximation from the state given by the observation.

        To make sure that `sum P = sum C` in this system, the **loads**
        are scaled up.

        This function can primarily be used to retrieve the active power
        in each branch of the grid.
        Parameters
        ----------
        obs : BaseObservation
            The observation (used to get the topology and the injections)

        Returns:
        - Boolean indicating if the power flow has converged.

        """

        # Update topological and injection parameters for the optimization model.
        self._update_topo_param(obs)
        self._update_inj_param(obs)

        # Define the optimization variables.
        theta = cp.Variable(shape=self.nb_max_bus)

        # Calculate the power flow on each line based on the difference of angles between origin and extremity.
        f_or = cp.multiply(
            1.0 / self._powerlines_x,
            (theta[self.bus_or.value] - theta[self.bus_ex.value]),
        )

        # Calculate the net power injection at each bus.
        inj_bus = self.load_per_bus - self.gen_per_bus

        # Ensure Kirchhoff's Current Law (KCL) is satisfied.
        KCL_eq = self._aux_compute_kcl(inj_bus, f_or)

        # find busbar that has no element connected to
        theta_is_zero = self._mask_theta_zero()

        # Define the optimization constraints.
        constraints = [theta[theta_is_zero] == 0] + [el == 0 for el in KCL_eq]

        # Define the optimization objective (minimize a constant to just satisfy the constraints).
        cost = 1.0

        # Solve the optimization problem with OSQP
        prob = cp.Problem(cp.Minimize(cost), constraints)
        has_converged = self._solve_problem(prob, solver_type=cp.OSQP)

        # If the problem has converged, store the computed power flows.
        if has_converged:
            self.flow_computed[:] = f_or.value
        else:
            logger.error(
                f"Problem diverged with dc approximation for all solver ({type(self).SOLVER_TYPES}). "
                "Is your grid connected (one single connex component) ?"
            )
            self.flow_computed[:] = np.NaN
        return has_converged

    def reset(self, obs: ObservationIDF2023):
        """
        Resets the agent's state at the beginning of an episode.

        Parameters:
        - obs: The current grid observation.
        """
        self._prev_por_error.value[:] = 0.0
        conv_ = self.run_dc(obs)
        if conv_:
            self._prev_por_error.value[:] = self.flow_computed - obs.p_or
        else:
            self.logger.warning(
                "Impossible to intialize the OptimCVXPY agent because the DC powerflow did not converge."
            )

    def compute_optimum_unsafe(self):
        # Initialize the CVXPY variables
        theta = cp.Variable(shape=self.nb_max_bus)
        curtailment_mw = cp.Variable(shape=self.nb_max_bus)
        storage = cp.Variable(shape=self.nb_max_bus)
        redispatching = cp.Variable(shape=self.nb_max_bus)

        # Create the CVXPY expressions
        f_or = cp.multiply(
            1.0 / self._powerlines_x,
            (theta[self.bus_or.value] - theta[self.bus_ex.value]),
        )
        f_or_corr = f_or - self._alpha_por_error * self._prev_por_error
        inj_bus = (self.load_per_bus + storage) - (
            self.gen_per_bus + redispatching - curtailment_mw
        )
        energy_added = (
            cp.sum(curtailment_mw)
            + cp.sum(storage)
            - cp.sum(redispatching)
            - self._storage_power_obs
        )

        # Define the constraints for the optimization problem
        # Theta constraints
        theta_constraints = [theta[self._mask_theta_zero()] == 0]

        # Kirchhoff Current Law constraints
        kcl_constraints = [el == 0 for el in self._aux_compute_kcl(inj_bus, f_or)]

        # Redispatching constraints
        redispatching_constraints = []
        for i in range(self.n_gen):
            bus_id = self.gen_to_subid[i]
            if self.env.gen_renewable[i]:
                redispatching_constraints.append(redispatching[bus_id] == 0)
            else:
                redispatching_constraints.append(
                    redispatching[bus_id] <= self.env.gen_max_ramp_up[i]
                )
                redispatching_constraints.append(
                    redispatching[bus_id] >= -self.env.gen_max_ramp_down[i]
                )

        # limit redispatching to possible values of redispatching
        redispatching_limit = [
            redispatching <= self.redisp_up,
            redispatching >= -self.redisp_down,
        ]

        # Curtailment constraints
        curtailment_constraints = [
            curtailment_mw <= self.curtail_up,
            curtailment_mw >= -self.curtail_down,
        ]

        # Storage constraints
        storage_constraints = [
            storage <= self.storage_up,
            storage >= -self.storage_down,
        ]

        # Energy constraints
        energy_constraints = [energy_added == 0]

        # Consolidating all constraints
        constraints = (
            theta_constraints
            + kcl_constraints
            + redispatching_constraints
            + redispatching_limit
            + curtailment_constraints
            + storage_constraints
            + energy_constraints
        )

        # Define the cost function
        cost = (
            self._penalty_curtailment_unsafe * cp.sum_squares(curtailment_mw)
            + self._penalty_storage_unsafe * cp.sum_squares(storage)
            + self._penalty_redispatching_unsafe * cp.sum_squares(redispatching)
            + cp.sum_squares(
                cp.pos(cp.abs(f_or_corr) - self._margin_th_limit * self._th_lim_mw)
            )
        )

        # Formulate the optimization problem
        prob = cp.Problem(cp.Minimize(cost), constraints)

        # Solve the optimization problem
        has_converged = self._solve_problem(prob, solver_type=cp.OSQP)

        # Extract the results or handle diverged case
        if has_converged:
            self.flow_computed[:] = f_or.value
            res = (curtailment_mw.value, storage.value, redispatching.value)
            self._storage_power_obs.value = 0.0
        else:
            # Handle the case when the optimization problem does not converge
            logger.error(
                "compute_optimum_unsafe: Problem diverged. No continuous action will be applied."
            )
            self.flow_computed[:] = np.NaN
            tmp_ = np.zeros(shape=self.nb_max_bus)
            res = (1.0 * tmp_, 1.0 * tmp_, 1.0 * tmp_)

        return res

    def compute_optimum_safe(self, obs: ObservationIDF2023, l_id=None):
        if l_id is not None:
            self.bus_ex.value[l_id] = obs.line_ex_to_subid[l_id]
            self.bus_or.value[l_id] = obs.line_or_to_subid[l_id]

        # Initialize the CVXPY variables
        theta = cp.Variable(shape=self.nb_max_bus)
        curtailment_mw = cp.Variable(shape=self.nb_max_bus)
        storage = cp.Variable(shape=self.nb_max_bus)
        redispatching = cp.Variable(shape=self.nb_max_bus)

        # Create the CVXPY expressions
        f_or = cp.multiply(
            1.0 / self._powerlines_x,
            (theta[self.bus_or.value] - theta[self.bus_ex.value]),
        )
        f_or_corr = f_or - self._alpha_por_error * self._prev_por_error
        inj_bus = (self.load_per_bus + storage) - (
            self.gen_per_bus + redispatching - curtailment_mw
        )
        energy_added = (
            cp.sum(curtailment_mw)
            + cp.sum(storage)
            - cp.sum(redispatching)
            - self._storage_power_obs
        )
        KCL_eq = self._aux_compute_kcl(inj_bus, f_or)
        theta_is_zero = self._mask_theta_zero()
        dispatch_after_this = self._past_dispatch + redispatching
        state_of_charge_after = self._past_state_of_charge + storage / (
            60.0 / obs.delta_time
        )

        # Define the constraints for the optimization problem
        constraints = (
            [theta[theta_is_zero] == 0]
            + [el == 0 for el in KCL_eq]
            + [f_or_corr <= self._margin_th_limit * self._th_lim_mw]
            + [f_or_corr >= -self._margin_th_limit * self._th_lim_mw]
            + [redispatching <= self.redisp_up, redispatching >= -self.redisp_down]
            + [curtailment_mw <= self.curtail_up, curtailment_mw >= -self.curtail_down]
            + [storage <= self.storage_up, storage >= -self.storage_down]
            + [energy_added == 0]
        )

        # Define the cost function
        cost = (
            self._penalty_curtailment_safe * cp.sum_squares(curtailment_mw)
            + self._penalty_storage_safe * cp.sum_squares(storage)
            + self._penalty_redispatching_safe * cp.sum_squares(redispatching)
            + self._weight_redisp_target * cp.sum_squares(dispatch_after_this)
            + self._weight_storage_target
            * cp.sum_squares(state_of_charge_after - self._storage_target_bus)
            + self._weight_curtail_target
            * cp.sum_squares(curtailment_mw + self.curtail_down)
        )

        # Formulate the optimization problem
        prob = cp.Problem(cp.Minimize(cost), constraints)

        # Extract the results or handle diverged case
        has_converged = self._solve_problem(prob, solver_type=cp.OSQP)
        if has_converged:
            self.flow_computed[:] = f_or.value
            res = (curtailment_mw.value, storage.value, redispatching.value)
            self._storage_power_obs.value = 0.0
        else:
            logger.error(
                "compute_optimum_safe: Problem diverged. No continuous action will be applied."
            )
            self.flow_computed[:] = np.NaN
            tmp_ = np.zeros(shape=self.nb_max_bus)
            res = (1.0 * tmp_, 1.0 * tmp_, 1.0 * tmp_)

        return res

    def _clean_vect(self, curtailment, storage, redispatching):
        """
        Sets small values of the provided vectors to zero to ensure sparsity.

        Parameters:
        - curtailment: Vector representing curtailment actions.
        - storage: Vector representing storage actions.
        - redispatching: Vector representing redispatching actions.
        """
        curtailment[np.abs(curtailment) < self.margin_sparse] = 0.0
        storage[np.abs(storage) < self.margin_sparse] = 0.0
        redispatching[np.abs(redispatching) < self.margin_sparse] = 0.0

    def to_grid2op(
        self,
        obs: ObservationIDF2023,
        curtailment: np.ndarray,
        storage: np.ndarray,
        redispatching: np.ndarray,
        base_action: BaseAction = None,
        safe=False,
    ) -> BaseAction:
        """
        Convert the optimization results into a valid Grid2Op action.

        Parameters:
        - obs: Current grid observation.
        - curtailment: Array representing the amount of generation to be curtailed.
        - storage: Array representing the storage actions.
        - redispatching: Array representing the redispatching actions.
        - base_action: An optional base action to start with.
        - safe: Boolean indicating whether we are in a safe mode or not.

        Returns:
        - A valid Grid2Op action.
        """

        # Clean the vectors to avoid small numerical issues.
        self._clean_vect(curtailment, storage, redispatching)

        # If no base action is provided, initialize an empty action.
        if base_action is None:
            base_action = self.action_space()

        # Update the action with storage decisions if there are any non-zero storage actions.
        if base_action.n_storage and np.any(np.abs(storage) > 0.0):
            storage_ = np.zeros(shape=base_action.n_storage)
            storage_[:] = storage[self.bus_storage.value]
            base_action.storage_p = storage_

        # Update the action with curtailment decisions, be carefull here,
        # the curtailment is given by the optimizer in the amount of MW you remove, grid2op expects a maximum value
        if np.any(np.abs(curtailment) > 0.0):
            curtailment_mw = np.zeros(shape=base_action.n_gen) - 1.0
            gen_curt = obs.gen_renewable & (obs.gen_p > 0.1)
            idx_gen = self.bus_gen.value[gen_curt]
            tmp_ = curtailment[idx_gen]
            modif_gen_optim = tmp_ != 0.0
            aux_ = curtailment_mw[gen_curt]
            aux_[modif_gen_optim] = (
                obs.gen_p[gen_curt][modif_gen_optim]
                - tmp_[modif_gen_optim]
                * obs.gen_p[gen_curt][modif_gen_optim]
                / self.gen_per_bus.value[idx_gen][modif_gen_optim]
            )
            aux_[~modif_gen_optim] = -1.0
            curtailment_mw[gen_curt] = aux_
            curtailment_mw[~gen_curt] = -1.0
            if safe:
                # id of the generators that are "curtailed" at their max value in safe mode i remove all curtailment
                gen_id_max = (
                    curtailment_mw >= obs.gen_p_before_curtail
                ) & obs.gen_renewable
                if np.any(gen_id_max):
                    curtailment_mw[gen_id_max] = base_action.gen_pmax[gen_id_max]
            base_action.curtail_mw = curtailment_mw

        # If in safe mode and no curtailment decisions have been made, reset all curtailment.
        elif safe and np.abs(self.curtail_down.value).max() == 0.0:
            vect = 1.0 * base_action.gen_pmax
            vect[~obs.gen_renewable] = -1.0
            base_action.curtail_mw = vect

        # Update the action with redispatching decisions.
        if np.any(np.abs(redispatching) > 0.0):
            redisp_ = np.zeros(obs.n_gen)
            gen_redi = obs.gen_redispatchable
            idx_gen = self.bus_gen.value[gen_redi]
            tmp_ = redispatching[idx_gen]
            redisp_avail = np.zeros(self.nb_max_bus)
            for bus_id in range(self.nb_max_bus):
                if redispatching[bus_id] > 0.0:
                    redisp_avail[bus_id] = obs.gen_margin_up[
                        self.bus_gen.value == bus_id
                    ].sum()
                elif redispatching[bus_id] < 0.0:
                    redisp_avail[bus_id] = obs.gen_margin_down[
                        self.bus_gen.value == bus_id
                    ].sum()
            prop_to_gen = np.zeros(obs.n_gen)
            redisp_up = np.zeros(obs.n_gen, dtype=bool)
            redisp_up[gen_redi] = tmp_ > 0.0
            prop_to_gen[redisp_up] = obs.gen_margin_up[redisp_up]
            redisp_down = np.zeros(obs.n_gen, dtype=bool)
            redisp_down[gen_redi] = tmp_ < 0.0
            prop_to_gen[redisp_down] = obs.gen_margin_down[redisp_down]

            # avoid numeric issues
            nothing_happens = (redisp_avail[idx_gen] == 0.0) & (
                prop_to_gen[gen_redi] == 0.0
            )
            set_to_one_nothing = 1.0 * redisp_avail[idx_gen]
            set_to_one_nothing[nothing_happens] = 1.0
            redisp_avail[idx_gen] = (
                set_to_one_nothing  # avoid 0. / 0. and python sends a warning
            )
            if np.any(np.abs(redisp_avail[idx_gen]) <= self.margin_sparse):
                logger.warning(
                    "Some generator have a dispatch assign to them by "
                    "the optimizer, but they don't have any margin. "
                    "The dispatch has been canceled (this was probably caused "
                    "by the optimizer not meeting certain constraints)."
                )
                this_fix_ = 1.0 * redisp_avail[idx_gen]
                too_small_here = np.abs(this_fix_) <= self.margin_sparse
                tmp_[too_small_here] = 0.0
                this_fix_[too_small_here] = 1.0
                redisp_avail[idx_gen] = this_fix_

            # Now I split the output of the optimization between the generators
            redisp_[gen_redi] = tmp_ * prop_to_gen[gen_redi] / redisp_avail[idx_gen]
            redisp_[~gen_redi] = 0.0
            base_action.redispatch = redisp_
        return base_action

    def reco_line(self, obs):
        # Determine the best line to reconnect by simulating the reconnection of each line.
        line_stat_s = obs.line_status
        cooldown = obs.time_before_cooldown_line
        maintenance = obs.time_next_maintenance
        can_be_reco = ~line_stat_s & (cooldown == 0) & (maintenance != 1)

        min_rho = np.inf
        action_chosen = None
        obs_simu_chosen = None
        if np.any(can_be_reco):
            actions = [
                self.action_space({"set_line_status": [(id_, +1)]})
                for id_ in np.where(can_be_reco)[0]
            ]
            for action in actions:
                obs_simu, _reward, _done, _info = obs.simulate(
                    action, time_step=self.time_step
                )
                if (obs_simu.rho.max() < min_rho) & (len(_info["exception"]) == 0):
                    action_chosen = action
                    obs_simu_chosen = obs_simu
                    min_rho = obs_simu.rho.max()
        return action_chosen, obs_simu_chosen

    def recover_reference_topology(self, observation, base_action, min_rho=None):
        # If no target loading ratio is provided, use the maximum observed loading ratio as the target.
        if min_rho is None:
            min_rho = observation.rho.max()

        action_chosen = None
        obs_chosen = None

        # Get the list of possible actions to revert the grid's topology to its reference state.
        ref_actions = self.action_space.get_back_to_ref_state(observation).get(
            "substation", None
        )

        if ref_actions is not None:
            for action in ref_actions:
                # Skip if the substation associated with the action is in cooldown mode.
                if (
                    observation.time_before_cooldown_sub[
                        int(action.as_dict()["set_bus_vect"]["modif_subs_id"][0])
                    ]
                    > 0
                ):
                    continue

                # Combine the base action with the current topology change action.
                combined_action = base_action + action
                obs_simu, _reward, _done, _info = observation.simulate(
                    combined_action, time_step=self.time_step
                )

                # If the simulated action results in an improvement (lower loading ratio) and no exceptions,
                # this action becomes the chosen one.
                if (obs_simu.rho.max() < min_rho) & (len(_info["exception"]) == 0):
                    action_chosen = action
                    obs_chosen = obs_simu
                    min_rho = obs_simu.rho.max()

        return action_chosen, obs_chosen, min_rho

    def change_substation_topology(self, observation, base_action, min_rho=None):
        # Decide which set of topology actions to consider based on the grid's state.
        if observation.rho.max() < self.rho_safe:
            if "N1_safe" in self.model:
                return self.get_topo_nn_act(observation, base_action, "N1_safe")
            topo_action = self.topo_actions_to_check_N1_safe
        elif observation.rho.max() > self.rho_danger:
            if all(observation.line_status):
                if "12_unsafe" in self.model:
                    return (
                        self.get_topo_nn_act(observation, base_action, "12_unsafe"),
                        observation,
                        0,
                    )
                topo_action = self.topo_actions_to_check_12_unsafe
            else:
                if "N1_unsafe" in self.model:
                    return self.get_topo_nn_act(observation, base_action, "N1_unsafe")
                topo_action = self.topo_actions_to_check_N1_unsafe
        else:
            if "N1_interm" in self.model:
                return self.get_topo_nn_act(observation, base_action, "N1_interm")
            topo_action = self.topo_actions_to_check_N1_interm

        # If no target loading ratio is provided, use the maximum observed loading ratio as the target. if min_rho is None:
        if min_rho is None:
            min_rho = observation.rho.max()

        action_chosen = None
        obs_chosen = None

        for action in topo_action:
            # Identify the substation associated with the action.
            substation_id = None
            for id_, k in enumerate(action._set_topo_vect):
                if k != 0:
                    _, _, substation_id = action._obj_caract_from_topo_id(id_)
                    break

            if substation_id is None:
                continue

            # Check if the substation is not in cooldown mode.
            if not observation.time_before_cooldown_sub[substation_id]:
                combined_action = base_action + action
                obs_simu, _reward, _done, _info = observation.simulate(
                    combined_action, time_step=self.time_step
                )

                # If the simulated action results in an improvement (lower loading ratio) and no exceptions,
                # this action becomes the chosen one.
                if (obs_simu.rho.max() < min_rho) & (len(_info["exception"]) == 0):
                    action_chosen = action
                    obs_chosen = obs_simu
                    min_rho = obs_simu.rho.max()

        return action_chosen, obs_chosen, min_rho

    def get_topo_nn_act(self, obs, base_action, action_space):
        gym_obs = self.gym_env[action_space].observation_space.to_gym(obs)
        gym_action, _states = self.model[action_space].predict(gym_obs)
        act = self.gym_env[action_space].action_space.from_gym(gym_action)
        combined_action = base_action + act
        obs_simu, _reward, _done, _info = obs.simulate(
            combined_action, time_step=self.time_step
        )
        return act, obs_simu, obs_simu.rho.max()

    def get_topology_action(self, observation, base_action, safe=False, min_rho=None):
        # If no target loading ratio is provided, use the maximum observed loading ratio as the target. if min_rho is None:
        if min_rho is None:
            min_rho = observation.rho.max()

        # First, try to revert to the reference topology.
        action_chosen, obs_chosen, rho_chosen = self.recover_reference_topology(
            observation, base_action, min_rho
        )

        # If the selected action from the above step improves the grid's state sufficiently, return it.
        if action_chosen is not None:
            if obs_chosen.rho.max() < 0.99:
                return action_chosen, obs_chosen

        # Otherwise, explore other topological configurations.
        action_chosen, obs_chosen, rho_chosen = self.change_substation_topology(
            observation, base_action, rho_chosen
        )

        if action_chosen is not None:
            return action_chosen, obs_chosen

        return None, None

    def recon_line_area(self, obs):
        """
        For each area, determine the best line to reconnect by simulating the reconnection of each line.

        Parameters:
        - obs: The current grid observation.

        Returns:
        - A list of chosen actions (one for each area).
        - A list of simulated observations resulting from these actions.
        """

        # Identify the lines that can be reconnected (i.e., currently disconnected and not in cooldown or maintenance).
        line_stat_s = obs.line_status
        cooldown = obs.time_before_cooldown_line
        maintenance = obs.time_next_maintenance
        can_be_reco = ~line_stat_s & (cooldown == 0)  # & (maintenance != 1)

        # Initialize variables to store best actions and their results for each area.
        min_rho_per_area = [np.inf for _ in self.lines_in_area]
        action_chosen_per_area = [
            self.action_space({}) for _ in self.lines_in_area
        ]  # Default to "do nothing" actions
        obs_simu_chosen_per_area = [None for _ in self.lines_in_area]

        # If there are lines that can be reconnected, simulate their reconnection.
        if np.any(can_be_reco):
            actions_and_ids = [
                (self.action_space({"set_line_status": [(id_, +1)]}), id_)
                for id_ in np.where(can_be_reco)[0]
            ]
            for action, id_ in actions_and_ids:
                obs_simu, _reward, _done, _info = obs.simulate(
                    action, time_step=self.time_step
                )

                # Determine the area of the line using lines_in_area.
                area_id = next(
                    (
                        idx
                        for idx, lines in enumerate(self.lines_in_area)
                        if id_ in lines
                    ),
                    None,
                )

                # If the simulation is successful and the result is better than previous simulations for the area,
                # store this action as the best action for that area.
                if area_id is not None and (
                    obs_simu.rho.max() < min_rho_per_area[area_id]
                ) & (len(_info["exception"]) == 0):
                    action_chosen_per_area[area_id] = action
                    obs_simu_chosen_per_area[area_id] = obs_simu
                    min_rho_per_area[area_id] = obs_simu.rho.max()

        return action_chosen_per_area, obs_simu_chosen_per_area

    def combine_actions(self, base_action, list_of_actions):
        """
        Combine a base action with a list of other actions.

        Parameters:
        - base_action: The primary action to be executed.
        - list_of_actions: A list of additional actions to be combined with the base action.

        Returns:
        - A combined action.
        """
        combined_action = base_action
        for action in list_of_actions:
            combined_action += action
        return combined_action

    def act(self, observation: ObservationIDF2023, reward=None, done: bool = False):
        """
        Decide the action to take given the current observation of the grid.

        Parameters:
        - observation: Current grid observation.
        - reward: Reward received from the previous action (unused in this function).
        - done: Boolean indicating if the episode is over.

        Returns:
        - A valid Grid2Op action.
        """

        # If the episode is over, return a 'do nothing' action.
        if done:
            return self.do_nothing

        # If it's the beginning of the episode, reset some internal state.
        if observation.current_step == 0:
            self.flow_computed[:] = np.NaN
            self._prev_por_error.value[:] = 0.0
        prev_ok = np.isfinite(self.flow_computed)

        # Update the previous flow error.
        self._prev_por_error.value[prev_ok] = np.minimum(
            self.flow_computed[prev_ok] - observation.p_or[prev_ok], 0.0
        )
        self._prev_por_error.value[~prev_ok] = 0.0

        # Initialize the action as 'do nothing'.
        act = self.action_space()

        # Simulate the effect of the 'do nothing' action.
        _obs = None
        _obs_simu, _reward, _done, _info = observation.simulate(act, time_step=1)
        if len(_info["exception"]) == 0:
            _obs = _obs_simu

        # Consider reconnecting lines based on a defined area or specific lines.
        if self.area:
            reco_act, reco_obs = self.recon_line_area(observation)
            if reco_act is not None:
                act = self.combine_actions(act, reco_act)
        else:
            reco_act, reco_obs = self.reco_line(observation)
            if reco_act is not None:
                act = act + reco_act
                _obs = reco_obs

        # Reset the computed flows.
        self.flow_computed[:] = np.NaN

        # If the grid's maximum line utilization is dangerous, take corrective actions.
        if observation.rho.max() > self.rho_danger:
            # Find the optimal topology action
            topo_act, topo_obs = self.get_topology_action(observation, act)
            if topo_act is not None:
                act = act + topo_act
                _obs = topo_obs

            # Update the observed storage power.
            self._update_storage_power_obs(observation)

            # Update internal parameters based on the new observation.
            if _obs is not None:
                self.update_parameters(_obs, safe=False)
            else:
                self.update_parameters(observation, safe=False)

            # Compute the optimal actions for curtailment, storage, and redispatching.
            curtailment, storage, redispatching = self.compute_optimum_unsafe()

            # Convert these decisions into a valid Grid2Op action.
            act = self.to_grid2op(
                observation,
                curtailment,
                storage,
                redispatching,
                base_action=act,
                safe=False,
            )

            # Find the best line to reconnect if needed.
            act = find_best_line_to_reconnect(observation, act)

            # Log a warning about the dangerous state.
            logger.warning(
                f"{[time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())]}: step {observation.current_step} with rho: {round(float(observation.rho.max()),2)}, unsafe -- disconnection line: {np.where(observation.rho == 0)[0]}"
            )

        elif observation.rho.max() < self.rho_safe:
            # if the topology can be reverted to its original state.
            action_array = revert_topo(self.action_space, observation)
            act = self.action_space.from_vect(action_array)
        else:
            # Update the observed storage power.
            self._update_storage_power_obs(observation)
            self.flow_computed[:] = observation.p_or
        return act


if __name__ == "__main__":
    import os

    from grid2op.MakeEnv import make
    from grid2op.Runner import Runner
    from lightsim2grid import LightSimBackend

    env = make("l2rpn_idf_2023", backend=LightSimBackend())
    agent = LJNAgent(env, env.action_space)
    runner = Runner(**env.get_params_for_runner(), agentClass=None, agentInstance=agent)

    runner.run(nb_episode=1, path_save="log/")
