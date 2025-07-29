import numpy as np
from crowd_sim.envs.utils.action import ActionXY
from crowd_sim.envs.policy.policy import Policy

class TurnRightAvoidance(Policy):
    def __init__(self):
        super().__init__()
        self.trainable = False
        self.kinematics = 'holonomic'
        self.multiagent_training = True
        self.turn_angle = np.pi / 18  # smaller incremental turn (10 degrees)
        self.current_vel = None  # will store current velocity vector

    def configure(self, config):
        pass

    def predict(self, state):
        self_state = state.self_state
        humans = state.human_states

        # Initialize current velocity toward goal if not set yet
        if self.current_vel is None:
            goal_vec = np.array([self_state.gx - self_state.px, self_state.gy - self_state.py])
            norm_goal = np.linalg.norm(goal_vec)
            if norm_goal > 0:
                self.current_vel = (goal_vec / norm_goal) * self_state.v_pref
            else:
                self.current_vel = np.array([0.0, 0.0])

        speed = self_state.v_pref
        vel = self.current_vel

        # Check for imminent collisions
        collision_risk = False
        for human_state in humans:
            rel_pos = np.array([human_state.px - self_state.px, human_state.py - self_state.py])
            dist = np.linalg.norm(rel_pos)
            if dist == 0:
                continue
            rel_vel = np.array([human_state.vx - vel[0], human_state.vy - vel[1]])
            rel_speed_sq = np.dot(rel_vel, rel_vel)
            if rel_speed_sq == 0:
                continue
            t_cpa = -np.dot(rel_pos, rel_vel) / rel_speed_sq
            if 0 < t_cpa < 3.0:
                cpa_pos = rel_pos + t_cpa * rel_vel
                cpa_dist = np.linalg.norm(cpa_pos)
                combined_radius = human_state.radius + self_state.radius + 0.2
                if cpa_dist < combined_radius:
                    collision_risk = True
                    break

        if collision_risk:
            # Rotate current velocity right by incremental angle
            cos_a = np.cos(-self.turn_angle)  # clockwise turn
            sin_a = np.sin(-self.turn_angle)
            vx_new = vel[0] * cos_a - vel[1] * sin_a
            vy_new = vel[0] * sin_a + vel[1] * cos_a
            vel = np.array([vx_new, vy_new])
        else:
            # No collision risk, reset velocity toward goal
            goal_vec = np.array([self_state.gx - self_state.px, self_state.gy - self_state.py])
            norm_goal = np.linalg.norm(goal_vec)
            if norm_goal > 0:
                vel = (goal_vec / norm_goal) * speed
            else:
                vel = np.array([0.0, 0.0])

        self.current_vel = vel  # store for next step

        return ActionXY(vel[0], vel[1])

