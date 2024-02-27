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
