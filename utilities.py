"""In this file all utility methods are collected which are jointly used by the teacher, tutor, junior and senior agent.
The major task of the methods is to communicate with the Grid2Op Environment.

Credit: Some of the methods are the enhanced methods of the original code, see
@https://github.com/AsprinChina/L2RPN_NIPS_2020_a_PPO_Solution
"""

import logging
import re
from copy import deepcopy
from typing import Iterator, List, Optional, Tuple

import grid2op
import numpy as np
from grid2op.Action import ActionSpace, BaseAction
from grid2op.dtypes import dt_int
from grid2op.Environment import BaseEnv
from grid2op.Observation import BaseObservation


def find_best_line_to_reconnect(
    obs: BaseObservation, original_action: BaseAction
) -> BaseAction:
    """Given an observation and action try to reconnect a line by modifying the given original action
    and returning the modified action with the reconnection if possible.

    Args:
        obs: The current observation of the agent.
        original_action: The action the agent is going to take.

    Returns:
        The modified original_action which tries to reconnect disconnected lines.

    """
    disconnected_lines = np.where(obs.line_status == False)[0]
    if len(disconnected_lines) == 0:
        return original_action
    min_rho = 10
    line_to_reconnect = -1
    for line in disconnected_lines:
        if not obs.time_before_cooldown_line[line]:
            reconnect_array = np.zeros_like(obs.rho, dtype=int)
            reconnect_array[line] = 1
            reconnect_action = deepcopy(original_action)
            reconnect_action.update({"set_line_status": reconnect_array})
            if not is_legal(reconnect_action, obs):
                continue
            o, _, done, info = obs.simulate(reconnect_action)
            if not is_valid(
                observation=obs, act=reconnect_action, done_sim=done, info_sim=info
            ):
                continue
            if o.rho.max() < min_rho:
                line_to_reconnect = line
                min_rho = o.rho.max()

    reconnect_out = deepcopy(original_action)
    if line_to_reconnect != -1:
        reconnect_array = np.zeros_like(obs.rho, dtype=int)
        reconnect_array[line_to_reconnect] = 1
        reconnect_out.update({"set_line_status": reconnect_array})

    return reconnect_out


def is_legal(action: BaseAction, obs: BaseObservation) -> bool:
    """Return true if the given action is valid under the current given observation.

    Args:
        action: The action to check for.
        obs: The current observation of the environment.

    Returns:
        Whether the given action is valid/legal or not.

    """

    def extract_line_number(s: str) -> int:
        match = re.search(r"_(\d+)$", s)
        if match:
            return int(match.group(1))
        else:
            raise ValueError(f"Unexpected key format: {s}")

    action_dict = action.as_dict()

    if action_dict == {}:
        return True

    topo_action_type = list(action_dict.keys())[0]
    legal_act = True

    # Check substations:
    if topo_action_type == "set_bus_vect" or topo_action_type == "change_bus_vect":
        substations = [
            int(sub) for sub in action.as_dict()[topo_action_type]["modif_subs_id"]
        ]

        for substation_to_operate in substations:
            if obs.time_before_cooldown_sub[substation_to_operate]:
                # substation is cooling down
                legal_act = False
            # Check lines:
            for line in [
                extract_line_number(key)
                for key, val in action.as_dict()[topo_action_type][
                    str(substation_to_operate)
                ].items()
                if "line" in val["type"]
            ]:
                if obs.time_before_cooldown_line[line] or not obs.line_status[line]:
                    # line is cooling down, or line is disconnected
                    legal_act = False
    elif topo_action_type == "set_line_status":

        lines = [
            int(line) for line in action.as_dict()[topo_action_type]["connected_id"]
        ]
        for line in lines:
            if obs.time_before_cooldown_line[line]:
                legal_act = False

    return legal_act


def check_convergence(action: BaseAction, obs: BaseObservation) -> bool:
    """Return true if the given action leads to convergence in the network under the current given observation.

    Args:
        action: The action to check for.
        obs: The current observation of the environment.

    Returns:
        Whether the given action leads to convergent state in the network or not.

    """

    simulator = obs.get_simulator()

    load_p_stressed = obs.load_p * 1.05
    gen_p_stressed = obs.gen_p * 1.05

    simulator_stressed = simulator.predict(
        act=action, new_gen_p=gen_p_stressed, new_load_p=load_p_stressed
    )

    if not simulator_stressed.converged:
        return False
    else:
        return True


def get_from_dict_set_bus(original: dict) -> dict:
    """Convert action from dictionary based on BaseAction.as_dict() to a dictionary that can be used
    as input for the action space.

    Args:
        original: Action in dictionary format.

    Returns:
        Dictionary with set_bus action

    """
    dict_act = {
        "lines_or_id": [],
        "lines_ex_id": [],
        "loads_id": [],
        "generators_id": [],
    }
    for key, value in original.items():
        for old, new in [
            ("line (origin)", "lines_or_id"),
            ("line (extremity)", "lines_ex_id"),
            ("load", "loads_id"),
            ("generator", "generators_id"),
        ]:
            if old == original[key]["type"]:
                dict_act[new].append((int(key), int(value["new_bus"])))

    return {"set_bus": dict_act}


def extract_action_set_from_actions(
    action_space: ActionSpace, action_vect: np.ndarray
) -> List[BaseAction]:
    """Method to separate multiple substation actions into single actions.

    This method is necessary to ensure that the tuple and triple actions are in accordance to
    the Grid2Op rules.

    Args:
        action_space: action space of Grid2Op environment.
        action_vect: single action from numpy array.

    Returns:
        List with actions.

    """
    action_set = []

    # Check if do nothing action:
    if not action_vect.any():
        return [action_space({})]

    # Convert into action:

    act_dict = action_space.from_vect(action_vect).as_dict()
    if "set_bus_vect" in act_dict.keys():
        act_t = act_dict["set_bus_vect"]

        # Get sub-ids
        changed_sub_ids = act_t["modif_subs_id"]

        if len(changed_sub_ids) > 1:
            # Collect single action
            for sub_id in changed_sub_ids:
                sub_action = action_space(get_from_dict_set_bus(act_t[sub_id]))

                action_set.append(sub_action)
            return action_set
        else:
            return [action_space.from_vect(action_vect)]  #

    if "change_bus_vect" in act_dict.keys():
        # We have an old action path with only change_bus actions
        # These actions are assumed to be unitary, thus we only return this action:
        act_t = act_dict["change_bus_vect"]
        if len(act_t["modif_subs_id"]) == 1:
            return [action_space.from_vect(action_vect)]
        else:
            raise NotImplementedError(
                "Multiple substations were modified in the change_bus action. This is not yet "
                "implemented in the tuple and triple approach. Please one use set_bus actions "
                "or unitary change_bus actions"
            )
    else:
        logging.warning(
            "Attention, a action was provided which could not be accounted for by the "
            "extract_action_set_from_actions method."
        )
        return [action_space.from_vect(action_vect)]


def split_action_and_return(
    obs: BaseObservation, action_space: ActionSpace, action_vect: np.ndarray
) -> Iterator[BaseAction]:
    """Split an action with potentially multiple affected stations and return them sequentially as a generator/iterator.

    Depending on the input, the method either executes the numpy array as a unitary step, or
    if the input requires multiple steps (i.e. in case of a tuple or triple action) the method
    orders the actions and selects the best choice, then executes the actions sequentially.

    Note that if multiple steps are computed only the last observation is return. Further, the
    reward then consists of the cumulative reward. As Example: The action is a triple action. Thus, the
    reward is the cumulative reward of all three steps.

    All actions are checked for line reconnect.

    Args:
        action_space: action space of Grid2Op environment.
        obs: Current Observation.
        action_vect: Teacher Action that can either be a unitary, tuple or triple action.

    Returns:
        The next best action to execute, without the line connect!!!

    """

    # Special case: Do nothing
    if not action_vect.any():
        yield action_space({})
        return

    # First extract action:
    split_actions = extract_action_set_from_actions(action_space, action_vect)
    # Now simulate through all actions:
    for _ in range(len(split_actions)):
        # Iterate through remaining actions and choose the best one to execute next
        obs_min = np.inf
        best_choice = None

        for act in split_actions:
            act_plus_reconnect = find_best_line_to_reconnect(obs, act)
            obs_f, _, done, info = obs.simulate(act_plus_reconnect)

            # Check if valid
            if not is_valid(
                observation=obs, act=act_plus_reconnect, done_sim=done, info_sim=info
            ):
                continue

            if obs_f.rho.max() < obs_min:
                best_choice = act
                obs_min = obs_f.rho.max()

        if best_choice is None:
            best_choice = action_space({})

        # Assert whether a reconnection of the lines might be
        yield best_choice
        if best_choice in split_actions:
            split_actions.remove(best_choice)


def is_valid(
    observation: BaseObservation,
    act: BaseAction,
    done_sim,
    info_sim,
    check_overload: Optional[bool] = False,
) -> bool:
    """Checks whether the simulation output is legal/valid for our configuration

    Args:
        observation: original observation
        act: action of the agent
        done_sim: done variable of the obs.simulate()
        info_sim: info variable of the obs.simulate()
        check_overload: Boolean, whether to simulate a stress of the generation and load

    Returns:
        bool: Whether the action is valid.

    """
    valid_action = True
    if done_sim:
        valid_action = False
    if not is_legal(act, observation):
        valid_action = False
    if info_sim["is_illegal"]:
        valid_action = False
    if info_sim["is_ambiguous"]:
        valid_action = False
    if any(info_sim["exception"]):
        valid_action = False
        # first check convergence of the stressed net after action
    if check_overload and not check_convergence(act, observation):
        valid_action = False
    return valid_action


def simulate_action(
    action_space: ActionSpace,
    obs: BaseObservation,
    action_vect: np.array,
    check_overload: Optional[bool] = False,
) -> Tuple[float, bool]:
    """Simulate the impact of the given action. This method can be used by both  the tutor and the
    Senior and should result in better choices, when tuple/tripple actions are implemented.

    Args:
        action_space: Action Space of the environment.
        obs: Current Observation of the Grid2Op environment.
        action_vect: Array that can be converted back to Grid2Op Actions. Tuples and Tripples are supported.
        check_overload: Boolean, whether to simulate a stress of the generation and load

    Returns:
        The max Obs of the action.

    """
    action = action_space.from_vect(action_vect)
    act_dict = action.as_dict()
    if "set_bus_vect" not in act_dict.keys():
        action = find_best_line_to_reconnect(obs=obs, original_action=action)
        obs_f, _, done, info = obs.simulate(action)
    elif len(act_dict["set_bus_vect"]["modif_subs_id"]) == 1:
        action = find_best_line_to_reconnect(obs=obs, original_action=action)
        obs_f, _, done, info = obs.simulate(action)
    else:
        # length is longer than 1:
        gen = split_action_and_return(
            obs=obs, action_space=action_space, action_vect=action_vect
        )
        action = next(gen)
        obs_f, _, done, info = obs.simulate(action)
    rho_max = obs_f.rho.max()

    valid_action = is_valid(
        observation=obs,
        act=action,
        done_sim=done,
        info_sim=info,
        check_overload=check_overload,
    )

    return rho_max, valid_action


def split_and_execute_action(
    env: BaseEnv, action_vect: np.ndarray
) -> Tuple[BaseObservation, float, bool, dict]:
    """Split and execute an action with potentially multiple affected stations.

    Depending on the input, the method either executes the numpy array as a unitary step, or
    if the input requires multiple steps (i.e. in case of a tuple or tripple action) the method
    does multiple steps.

    Note that if multiple steps are computed only the last observation is return. Further, the
    reward then consists of the cummulative reward. As Example: The action is a tripple action. Thus, the
    reward is the cummulative reward of all three steps.

    All actions are checked for line reconnect.

    Args:
        env: Grid2Op Environment.
        action_vect: Teacher Action that can either be a unitary, tuple or tripple action.

    Returns: Output of the env.step() function consisting of an Observation, (cumulative) Reward, a Done statement
    and info.

    """

    # First extract action:
    split_actions = extract_action_set_from_actions(env.action_space, action_vect)
    # Now simulate through all actions:
    cum_rew = 0
    done = False
    obs = env.get_obs()
    info = {}
    for _ in range(len(split_actions)):
        # Iterate through remaining actions and choose the best one to execute next
        obs_min = np.inf
        best_choice = find_best_line_to_reconnect(obs, env.action_space({}))

        chosen_act = None
        for act in split_actions:
            act_plus_reconnect = find_best_line_to_reconnect(obs, act)
            if not is_legal(act_plus_reconnect, obs):
                continue

            obs_, _, done, _ = obs.simulate(act_plus_reconnect)
            if obs_.rho.max() < obs_min and done is False:
                best_choice = act_plus_reconnect
                chosen_act = act
                obs_min = obs_.rho.max()

        # Assert whether a reconnection of the lines might be

        obs, reward, done, info = env.step(best_choice)
        cum_rew += reward
        if done:
            break
        if chosen_act:
            split_actions.remove(chosen_act)

    return obs, cum_rew, done, info


def revert_topo(
    action_space: grid2op.Action.ActionSpace,
    obs: grid2op.Observation.BaseObservation,
    rho_limit: Optional[float] = 0.8,
) -> np.array:
    """Method, if the topology can be reverted to its original state.

    Given that the original state is all substations at bus bar 1, we search for any bus bars equal to
    2. If one is found, we check whether a reversion of the substation reduces the rho.max(). To
    ensure that the minimization only happens, when the grid is stable, we propose a
    rho_limit. If the rho.max() is above the limit, no reversion takes place!

    Args:
        action_space: Action Space of the environment.
        obs: Current Observation of the Grid2Op Environment.
        rho_limit: Optional Limit for the topology search.

    Returns:
        Array of topo action.

    """
    act_best, obs_sim, idx = None, None, None
    if np.any(obs.topo_vect == 2) and np.all(obs.line_status):
        available_actions = {}
        for sub_id in range(obs.n_sub):
            if np.any(obs.sub_topology(sub_id) == 2):
                available_actions[sub_id] = action_space(
                    {
                        "set_bus": {
                            "substations_id": [
                                (
                                    sub_id,
                                    np.ones(
                                        len(obs.sub_topology(sub_id)), dtype=dt_int
                                    ),
                                )
                            ]
                        }
                    }
                ).to_vect()

        if any(available_actions):
            min_rho = min([obs.rho.max(), rho_limit])

            for sub_id, act_a in available_actions.items():
                obs_sim, valid_action = simulate_action(
                    action_vect=act_a, action_space=action_space, obs=obs
                )
                if not valid_action:
                    continue
                if obs_sim < min_rho:
                    idx = sub_id
                    act_best = act_a.copy()

    if act_best is not None:
        logging.info(
            f"Revert Substation {idx} from {available_actions.keys()} to original with {obs.rho.max()} and"
            f" {obs_sim}"
        )
        return act_best
    else:
        return action_space({}).to_vect()


def map_actions(list_of_actions: List[np.ndarray]) -> List[dict]:
    """This method iterates over the list of actions and passes an id for all actions. This helps for
    the junior and the senior evaluation.

    Args:
        list_of_actions: List that contains the actions in numpy format.

    Returns:
        List that contains a dictionary with the id as key and the action as value.

    """
    total_action = []
    idx = 0
    for actions in list_of_actions:
        act_id = {}
        for i, act in enumerate(actions):
            act_id[i + idx] = act
        idx += len(act_id)
        total_action.append(act_id)

    return total_action
