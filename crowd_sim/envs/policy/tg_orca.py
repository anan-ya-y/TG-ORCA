import numpy as np
from crowd_sim.envs.policy.policy import Policy
from crowd_sim.envs.utils.action import ActionXY


class TG_ORCA(Policy):
    def __init__(self):
        super().__init__()
        self.name = 'TG_ORCA'
        self.trainable = False

    def configure(self, config):
        assert True

    def predict(self, state):
        self_state = state.self_state
        theta = np.arctan2(self_state.gy-self_state.py, self_state.gx-self_state.px)
        vx = np.cos(theta) * self_state.v_pref
        vy = np.sin(theta) * self_state.v_pref
        action = ActionXY(vx, vy)

        return action