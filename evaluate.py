import os
import warnings

from grid2op.MakeEnv import make
from grid2op.Runner import Runner
from l2rpn_baselines.utils.save_log_gif import save_log_gif
from lightsim2grid import LightSimBackend

from l2rpn_baselines.LJNAgent import LJNAgent


def evaluate(
    env,
    load_path=".",
    logs_path=None,
    nb_episode=1,
    nb_process=1,
    max_steps=-1,
    verbose=True,
    save_gif=False,
    **kwargs
):
    """
    Evaluate the agent with a default config based on l2rpn_idf_2023 environment

    Parameters
    ----------
    env: :class:`grid2op.Environment.Environment`
        The environment on which the baseline will be evaluated.

    load_path: ``str``
        The path where the model is stored. This is used by the agent when calling "agent.load)

    logs_path: ``str``
        The path where the agents results will be stored.

    nb_episode: ``int``
        Number of episodes to run for the assessment of the performance.
        By default it's 1.

    nb_process: ``int``
        Number of process to be used for the assessment of the performance.
        Should be an integer greater than 1. By defaults it's 1.

    max_steps: ``int``
        Maximum number of timestep each episode can last. It should be a positive integer or -1.
        -1 means that the entire episode is run (until the chronics is out of data or until a game over).
        By default it's -1.

    verbose: ``bool``
        verbosity of the output

    save_gif: ``bool``
        Whether or not to save a gif into each episode folder corresponding to the representation of the said episode.

    kwargs:
        Other key words arguments that you are free to use for either building the agent save it etc.

    Returns
    -------
    ``None``
    """

    runner_params = env.get_params_for_runner()
    runner_params["verbose"] = verbose

    # Create the agent - using default parameters suitable for l2rpn_idf_2023
    agent = LJNAgent(
        env,
        env.action_space,
        verbose=verbose,
    )

    # Build the runner
    runner = Runner(**env.get_params_for_runner(), agentClass=None, agentInstance=agent)

    if logs_path is not None:
        os.makedirs(logs_path, exist_ok=True)
    results = runner.run(
        path_save=logs_path,
        nb_episode=nb_episode,
        nb_process=nb_process,
        max_iter=max_steps,
        pbar=verbose,
    )

    print("Evaluation summary:")
    for _, chron_name, cum_reward, nb_time_step, max_ts in results:
        msg_tmp = "chronics at: {}".format(chron_name)
        msg_tmp += "\ttotal score: {:.6f}".format(cum_reward)
        msg_tmp += "\ttime steps: {:.0f}/{:.0f}".format(nb_time_step, max_ts)
        print(msg_tmp)

    if save_gif:
        save_log_gif(logs_path, results)


if __name__ == "__main__":
    import grid2op
    from l2rpn_baselines.utils import cli_eval
    from lightsim2grid.lightSimBackend import LightSimBackend

    args_cli = cli_eval().parse_args()
    env = grid2op.make("l2rpn_idf_2023", backend=LightSimBackend())
    print("--- Starting evaluation on l2rpn_idf_2023 ---")
    evaluate(
        env,
        load_path="./",
        nb_episode=args_cli.nb_episode,
        nb_process=args_cli.nb_process,
        max_steps=args_cli.max_steps,
        verbose=args_cli.verbose,
        save_gif=args_cli.save_gif,
    )
