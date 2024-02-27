import logging
import re
from multiprocessing import Pool

import grid2op
from grid2op.Chronics import MultifolderWithCache
from grid2op.Runner import Runner
from lightsim2grid.lightSimBackend import LightSimBackend
from tqdm import tqdm

from .LJNAgent import LJNAgent
from .train.utils import TRAIN_ENV_NAME


def run_expert_agent(id: int):
    """This function use the grid2op runner on a single core with the training env

    Parameters
    ----------
    id : int
        index for multi-processing purpose
    """
    env = grid2op.make(
        TRAIN_ENV_NAME, backend=LightSimBackend(), chronics_class=MultifolderWithCache
    )
    env.chronics_handler.real_data.set_filter(
        lambda x: re.match(f".*_{id}$", x) is not None
    )
    env.chronics_handler.real_data.reset()

    NB_EPISODE = 52  # len(os.listdir('data/l2rpn_idf_2023_train/chronics'))
    NB_CORE = 1
    logging.info(f"Starting runner on {NB_EPISODE} episodes on {NB_CORE}")
    PATH_SAVE = f"agents_log_BC/"  # and store the results in the "agents_log" folder

    # initilize agent
    agent = LJNAgent(env, env.action_space)
    agent.id_filter = id
    # use the runner
    runner = Runner(**env.get_params_for_runner(), agentClass=None, agentInstance=agent)
    runner.run(nb_episode=NB_EPISODE, nb_process=NB_CORE, path_save=PATH_SAVE)
    agent.reset(env.reset())
    logging.info("Done")


if __name__ == "__main__":
    ## Using multi-processing here instead of multiple processes with the runner because of a bug ##
    pool = Pool(processes=16)
    pool.map(run_expert_agent, range(16))
