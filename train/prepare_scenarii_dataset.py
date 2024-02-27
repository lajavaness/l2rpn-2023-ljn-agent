import logging
import os
import random
from pathlib import Path

import grid2op
import pandas as pd
from grid2op.utils import ScoreL2RPN2023
from lightsim2grid.lightSimBackend import LightSimBackend

from .utils import CHALLENGE_ENV, DATA_DIR, SEED_VALUE


def prepare_dataset_and_scoring(env_name: str = CHALLENGE_ENV, val_ratio: float = 0.10):
    """
    This function prepares the dataset and scoring statistics.
    It performs a train-test split of the data based on the env name provided and ratio.

    Parameters
    ----------
    env_name : str, optional
        The str id of grid2op Env, by default CHALLENGE_ENV : l2rpn_idf_2023
    val_ratio : float, optional
        The ratio of the data to be used for test set, by default 0.10
    """
    grid2op.change_local_dir(DATA_DIR)
    logging.info(f"Loading parent env : {env_name} from {DATA_DIR}")
    env = grid2op.make(env_name, backend=LightSimBackend())

    if not Path(DATA_DIR, f"{env_name}_val").exists():
        chronics_list = os.listdir(os.path.join(DATA_DIR, CHALLENGE_ENV, "chronics"))
        chronics_list = [x.split("_")[0] for x in chronics_list]
        chronics_list = pd.DataFrame(
            [
                {"month": int(x.split("-")[1]), "day": int(x.split("-")[2])}
                for x in chronics_list
            ]
        )
        logging.info(chronics_list)
        val_id = []
        for i in range(1, 13):
            chronics = chronics_list[chronics_list.month == i]
            nb_weeks = len(chronics)
            nb_val = int(round(val_ratio * nb_weeks))
            days = random.choices(chronics.day.unique(), k=nb_val)
            ids = random.sample(range(16), k=nb_val)
            for j in range(nb_val):
                val_id.append("2035-%02d-%02d_%s" % (i, days[j], ids[j]))
        logging.info(val_id)
        env.train_val_split(val_id)
    val_set_dir = os.listdir(os.path.join(DATA_DIR, env_name + "_val", "chronics"))
    nb_episode = len(val_set_dir)
    logging.info(nb_episode)
    val_env = grid2op.make(env_name + "_val", backend=LightSimBackend())
    logging.info("Preparing scoring for validation set. This might take a while.")
    score = ScoreL2RPN2023(
        val_env,
        nb_scenario=nb_episode,
        env_seeds=[SEED_VALUE for _ in range(nb_episode)],
        agent_seeds=[SEED_VALUE for _ in range(nb_episode)],
        nb_process_stats=1,
        weight_op_score=0.8,
        weight_assistant_score=0.0,
        weight_nres_score=0.2,
        verbose=1,
    )

    logging.info("Statistics ready for scoring")


if __name__ == "__main__":
    prepare_dataset_and_scoring()
