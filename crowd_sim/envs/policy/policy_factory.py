from crowd_sim.envs.policy.linear import Linear
from crowd_sim.envs.policy.orca import ORCA
from crowd_sim.envs.policy.tg_orca import TG_ORCA
from crowd_sim.envs.policy.turn_right_avoidance import TurnRightAvoidance

def none_policy():
    return None


policy_factory = dict()
policy_factory['linear'] = Linear
policy_factory['orca'] = ORCA
policy_factory['tg_orca'] = TG_ORCA
policy_factory['tra'] = TurnRightAvoidance
policy_factory['none'] = none_policy
