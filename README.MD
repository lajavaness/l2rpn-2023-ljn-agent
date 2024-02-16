# LJN Agent - A modular agent for combined continuous control and topology management on large grid

[![Static Badge](https://img.shields.io/badge/La_Javaness-blue?style=flat&labelColor=FFFFFF&color=0000FF)](https://www.lajavaness.com/)
![Static Badge](https://img.shields.io/badge/license-MPL%202.0-black)

This agent is a simplified version of the winning agent of the L2RPN 2023 IDF AI challenge co-organized by RTE and Paris Region. It comes as an enhanced combination inspired by pre-existing baselines such as the curriculum agent and the optimCVXPY agent with additional features. Its development was guided towards the challenge's specific requirements. Especially, it combines continuous control and bus reconfiguration decision making with additional heuristics to operate the grid safely at low cost (i.e. maximizing an operational score) while maximizing the use of renewable energy available at all time.

This agent is highly competitive on large complex environment while remaining computationally efficient (max. decision time < 3 seconds).


**Content**
1. [Authors](#authors)
2. [Setup](#setup)
3. [How to Run the Agent](#how-to-run-the-agent)
   - [Training on custom environment](#training-on-a-custom-environment)
   - [Evaluate](#evaluate)
   - [Parameters](#parameters)
4. [Acknowledgments](#acknowledgments)
5. [License](#license)

## Authors

- Van Tuan DANG, Phd, Data-Scientist at La Javaness - [van.tuan\@lajavaness.com](mailto:van.tuan@lajavaness.com?subject=Test) [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/van-tuan-dang-7b0b4444/)
- Jules SINTES, Data-Scientist at La Javaness - [jules\@lajavaness.com](mailto:jules@lajavaness.com?subject=Test) [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/jules-sintes/)

## Setup

The python package CVXPY is required to run the agent. It can be installed through `pip install grid2op[optional]` or simply `pip install CVXPY`.
This agent has been developed for the particular environment of the L2RPN Ile-de-France challenge 2023, hence the provided action set is based on the environment with id `l2rpn_idf_2023` and actions have been cherry-picked for this particular setup aiming at optimizing the associated scoring system.


## How to run the agent

### Training on a custom environment

The training process to cherry pick the discrete bus reconfiguration mainly relies on exhaustive simulation and greedy search on the complete action space. It mainly used the work done for the [curriculum agent](https://github.com/FraunhoferIEE/curriculumagent/tree/master) with custom implementations for the specific needs within the frame of the L2RPN IDF 2023 challenge. This process is computionally intensive and it is recommended to use large CPU resources when trying to generate action space on large environment. 

The agent requires 4 different action spaces described as follow :
| Name | Description | Size for l2rpn_idf_2023 |
|---|---|---|
| action_12_unsafe | The grid is in unsafe state with no line disconnection | 421 |
| action_N1_interm | The grid is in an untermediate state and there is a line disconnection | 136 |
| action_N1_unsafe | The grid is in a unsafe state and there is a line disconnection | 909 |
| action_N1_safe | The grid is in a safe state and there is a line disconnection | 50 |

_NOTE : The size of the provided actions spaces are given as an indicator to build a custom action space for a different environment. The complete size of the action space of bus reconfiguration is _70k_ unitary actions for this environment. The actions are encoded as vectors, as implemented in the grid2op package._

### Evaluate

Given the provided action space, the evaluation should be run on environment `l2rpn_idf_2023` but one can try with its own action space set on a different environment.

**Command line** 
```bash
python l2rpn_baselines/LJNAgent/evaluate.py --nb_episode=2 --nb_process=1 --verbose=True
```

**Python code**
```python
from l2rpn_baselines.LJNAgent import LJNAgent, evaluate
from lightsim2grid.lightSimBackend import LightSimBackend # Recommended for faster simulation
from grid2op import make

env = make("l2rpn_idf_2023", backend = LightSimBackend())
evaluate(env,
        logs_path=None,
        nb_episode=10,
        nb_process=1,
        max_steps=-1,
        verbose=True,
        save_gif=False)

```

### Parameters

#### High Level heuristics

| Parameter                      | Value                      | Description                                   |
| ------------------------------ | -------------------------- | --------------------------------------------- |
| rho_danger                     | 0.99                       | Line capacity danger threshold               |
| rho_safe                       | 0.9                        | Line capacity safe state threshold           |
| areas                          | True                       | Flag for considering areas                   |
| sim_range_time_step            | 1                          | Number of time-step to simulate when checking action choice|

#### Continuous opitmization

| Parameter                      | Value                      | Description                                   |
| ------------------------------ | -------------------------- | --------------------------------------------- |
| margin_th_limit                | 0.93                       | Margin thermal limit                         |
| penalty_curtailment_unsafe     | 15                         | Penalty for curtailment in unsafe conditions |
| penalty_redispatching_unsafe   | 0.005                      | Penalty for redispatching in unsafe conditions|
| penalty_storage_unsafe         | 0.0075                     | Penalty for storage in unsafe conditions     |
| penalty_curtailment_safe       | 0.0                        | Penalty for curtailment in safe conditions   |
| penalty_redispatching_safe     | 0.0                        | Penalty for redispatching in safe conditions |
| penalty_storage_safe           | 0.0                        | Penalty for storage in safe conditions       |
| weight_redisp_target           | 1.0                        | Weight for redispatching target               |
| weight_storage_target          | 1.0                        | Weight for storage target                     |
| weight_curtail_target          | 1.0                        | Weight for curtailment target                 |
| margin_rounding                | 0.01                       | Margin rounding                              |
| margin_sparse                  | 5e-3                       | Sparse margin                                |
| max_iter                       | 100000                     | Maximum number of iterations of the solver   |


#### Bus reconfiguration

There is a tradeoff between computationnal efficiency and size of action space: Larger - _high-quality_ - action space should increase agent's performance while the decision time might significantly decrease. However, we found out that small cherry-picked action spaces are well suited to handle most of the situations.

## Reference

This baseline agent is mainly inspired by the work done on :
- [The OptimCVXPY agent from this package](https://github.com/rte-france/l2rpn-baselines/tree/master/l2rpn_baselines/OptimCVXPY)
- [Binbinchen's solution to the NEURIPS 2020 challenge](https://github.com/AsprinChina/L2RPN_NIPS_2020_a_PPO_Solution)
- [The curriculum agent by Fraunhofer Institute](https://github.com/FraunhoferIEE/curriculumagent)

## License

This project is licensed under the Mozilla Public License 2.0 - see the [LICENSE](LICENSE) file for details.
