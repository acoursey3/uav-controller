import numpy as np
import torch
from stable_baselines3 import PPO

from lstm import LSTM
from pid_controller import PIDController, PID_Params
import helper

class DisturbanceRejectionController:
    def __init__(self, starting_pos:np.ndarray, use_gpu:bool=True):
        if use_gpu:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = "cpu"

        pid_params = PID_Params()
        self.max_vel = 15

        self.disturbance_estimator = self.load_lstm()
        self.rl_agent = self.load_rl()
        self.pos_pid = PIDController(k_p=pid_params.pos_p, k_i=pid_params.pos_i, k_d=pid_params.pos_d, max_err_i=0)
        self.prev_pos = starting_pos

        self.lstm_buffer = [np.zeros(9)] * 5
        self.t = 0

    def control(self, pos: np.ndarray, vel: np.ndarray, eul: np.ndarray, rates: np.ndarray, next_wp: np.ndarray, prev_wp: np.ndarray):
        """
        Determine the reference velocity and change in reference velocity given by the disturbance rejection controller.

        Parameters
        ----------
        pos : np.ndarray
            The X, Y, Z position of the UAV in meters from the starting point (ENU frame).
        vel : np.ndarray
            The inertial frame velocity of the UAV.
        eul : np.ndarray
            The roll, pitch, yaw in radians.
        rates : np.ndarray
            The roll, pitch, and yaw angular rates.
        next_wp : np.ndarray
            The X, Y, Z coordinates of the next waypoint.
        prev_wp : np.ndarray
            The X, Y, Z coordinates of the previously-reached waypoint.

        Returns
        -------
        The reference velocity in the inertial frame: np.ndarray, the modification to that velocity: np.ndarray
        """
        # Only use the RL agent every 0.5 seconds
        # Assuming data comes every 0.25 seconds, the rate the LSTM and PID position controller operate
        predict_rl = self.t % 2 == 0
        
        dcm = helper.direction_cosine_matrix(eul[0], eul[1], eul[2])
        body_vel = helper.inertial_to_body(vel, dcm)

        str_err = helper.calc_intersection_distance(prev_wp, next_wp, pos)
        wp_err = pos - next_wp

        ref_vel = self.compute_ref_vel(pos, str_err, wp_err, eul, self.pos_pid)

        curr_lstm_input = helper.normalize_lstm(np.concatenate([(pos - self.prev_pos), body_vel, eul]))
        self.prev_pos = pos
        self.lstm_buffer.append(curr_lstm_input)
        self.lstm_buffer.pop(0)

        if predict_rl:
            self.delta_vel = self.compute_delta_vel(self.lstm_buffer, np.concatenate([wp_err, body_vel, eul, rates, str_err]))

        return ref_vel, self.delta_vel

    def compute_ref_vel(self, pos, str_err, wp_err, eul, pos_pid):
        intersection_point = pos - str_err 
        next_waypt = pos - wp_err
        ref_pos = helper.calculate_safe_sliding_bound(next_waypt, intersection_point, distance=23)
        inert_ref_vel = self.pos_controller(ref_pos, pos, pos_pid)
        inert_ref_vel = np.clip(inert_ref_vel, -self.max_vel, self.max_vel)
        inert_ref_vel_leashed = helper.vel_leash(inert_ref_vel, eul, self.max_vel)
        inert_ref_vel_leashed = inert_ref_vel_leashed

        return np.array([inert_ref_vel_leashed[0], inert_ref_vel_leashed[1], inert_ref_vel[2]])

    def compute_delta_vel(self, lstm_input, rl_input):
        disturbance_pred = self.disturbance_estimator(torch.Tensor(np.array(lstm_input)).unsqueeze(0).to(self.device)).cpu().detach().numpy()[0]
        rl_state = np.concatenate([rl_input, disturbance_pred[0:2]])
        rl_state = helper.normalize_rl(rl_state)
        action = self.rl_agent.predict(rl_state, deterministic=True)[0]
        return action

    def pos_controller(self, ref_pos, pos, pos_pid):
        ref_vel = pos_pid.step(ref_pos, pos, dt=0.25)
        return ref_vel

    def load_lstm(self):
        lstm = LSTM(9, 64, 2, 3).to(self.device)
        lstm.load_state_dict(torch.load('../saved_models/lstm_disturbance_4hz.pth'))
        return lstm

    def load_rl(self):
        agent = PPO.load('../saved_models/rl_agent.zip') 
        return agent 

