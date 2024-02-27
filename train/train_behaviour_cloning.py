import os

import datasets
import numpy as np
import torch as th
from grid2op.Chronics import MultifolderWithCache
from imitation.algorithms import bc
from imitation.data.huggingface_utils import TrajectoryDatasetSequence
from imitation.util.util import save_policy
from lightsim2grid.lightSimBackend import LightSimBackend
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy

from .env_components.custom_rewards import PPO_Reward
from .utils import ACT_SPACE_MAP, ASSET_DIR, TRAIN_ENV_NAME, build_gym_env


class FeedForward1024Policy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, net_arch=[1024, 1024])


AGENT_LOG_PATH = "./agents_log_BC"
DATASET_PATH = "./expert_datasets"
ASSET_DIR = "./l2rpn-2023-ljn-agent/action"


def train_BC(action_space_name: str, model_folder_path: str, n_epochs: int = 20):
    """
    Train policy on specified action space using Behaviour Cloning algorithm

    Parameters
    ----------
    action_space_name : str
        Name of the trained action space this should match the name of a stored action space.
    model_folder_path : str
        Output folder - If it does not exist it will be created.
    n_epochs : int
        Number of training epochs
    """
    action_space_path = os.path.join(ASSET_DIR, action_space_name + ".npz")
    env_gym, action_space = build_gym_env(
        TRAIN_ENV_NAME,
        action_space_path,
        backend=LightSimBackend(),
        reward_class=PPO_Reward,
        chronics_class=MultifolderWithCache,
    )

    if not os.path.exists(model_folder_path):
        os.mkdir(model_folder_path)

    dataset_path = os.path.join(DATASET_PATH, action_space_name)
    dataset = datasets.load_from_disk(os.path.join(dataset_path))
    transitions = TrajectoryDatasetSequence(
        dataset,
    )

    rng = np.random.default_rng(0)

    bc_trainer = bc.BC(
        observation_space=env_gym.observation_space,
        action_space=action_space,
        demonstrations=transitions,
        rng=rng,
        policy=FeedForward1024Policy(
            observation_space=env_gym.observation_space,
            action_space=env_gym.action_space,
            # Set lr_schedule to max value to force error if policy.optimizer
            # is used by mistake (should use self.optimizer instead).
            lr_schedule=lambda _: th.finfo(th.float32).max,
        ),
    )
    bc_trainer.train(n_epochs=n_epochs)

    save_policy(
        bc_trainer.policy,
        os.path.join(model_folder_path, f"model_bc_{action_space_name}.zip"),
    )


def train_PPO(
    action_space_name: str,
    model_folder_path: str,
    policy_class: ActorCriticPolicy = FeedForward1024Policy,
    total_timesteps: int = 1e5,
):
    """
    Use pre-trained policy for further RL training. To be used after train_BC function

    Parameters
    ----------
    action_space_name : str
        Name of the trained action space this should match the name of a stored action space and a model trained with behaviour cloning.
    model_folder_path : str
        Path to the saved model trained with behaviour cloning
    policy_class : ActorCriticPolicy, optional
        The Actor Critic policy architecture to use, by default FeedForward1024Policy
    total_timesteps : int, optional
        Number of training steps, by default 1e5
    """
    action_space_path = os.path.join(ASSET_DIR, action_space_name + ".npz")
    env_gym, action_space = build_gym_env(
        TRAIN_ENV_NAME,
        action_space_path,
        backend=LightSimBackend(),
        reward_class=PPO_Reward,
        chronics_class=MultifolderWithCache,
    )
    model = PPO(policy_class, env_gym)
    model.policy = bc.reconstruct_policy(
        os.path.join(model_folder_path, f"model_bc_{action_space_name}.zip")
    )
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    model.save(os.path.join(model_folder_path, f"model_rl_{action_space_name}.zip"))


if __name__ == "__main__":
    model_path = "bc_models"
    model_path_PPO = "ppo_models"
    for act_space in ACT_SPACE_MAP.values():
        train_BC(act_space, model_path, n_epochs=15)
        train_PPO(act_space, model_path, total_timesteps=2e5)
