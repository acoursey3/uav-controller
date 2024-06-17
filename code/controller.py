import numpy as np
import torch

from lstm import LSTM
from pid_controller import PIDController, PID_Params
import helper

class DisturbanceRejectionController:
    def __init__(starting_pos:np.ndarray, use_gpu:bool=True):
        pid_params = PID_Params()
        self.max_vel = 15

        self.disturbance_estimator = load_lstm()
        self.rl_agent = load_rl()
        self.pos_pid = PIDController(k_p=pid_params.pos_p, k_i=pid_params.pos_i, k_d=pid_params.pos_d, max_err_i=0)
        self.previous_pos = starting_pos

        self.lstm_buffer = [np.zeros(9)] * 5
        self.t = 0

        if use_gpu:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = "cpu"

    def control(self, pos: np.ndarray, vel: np.ndarray, eul: np.ndarray, rates: np.ndarray, next_wp: np.ndarray, prev_wp: np.ndarray):
        """
        Determine the reference velocity and change in reference velocity given by the disturbance rejection controller.

        Parameters
        ----------
        pos : np.ndarray
            The X, Y, Z position of the UAV in meters from the starting point (ENU frame).
        vel : np.ndarray
            The inertial velocity of the UAV.
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
        The reference velocity: np.ndarray, the modification to that velocity: np.ndarray
        """
        # Only use the RL agent every 0.5 seconds
        # Assuming data comes every 0.25 seconds, the rate the LSTM and PID position controller operate
        predict_rl = self.t % 2 == 0

        str_err = helper.calc_intersection_distance(prev_wp, next_wp, pos)
        wp_err = next_wp - pos

        ref_vel = compute_ref_vel(pos, str_err, wp_err, eul, self.pos_pid)

        curr_lstm_input = normalize_lstm(np.concatenate([(pos - self.prev_pos), vel, eul]))
        self.previous_pos = pos
        self.lstm_buffer.append(curr_lstm_input)
        self.lstm_buffer.pop(0)

        if predict_rl:
            self.delta_vel = compute_delta_vel(self.lstm_buffer, rl_input)

        return ref_vel, self.delta_vel

    def compute_ref_vel(self, pos, str_err, wp_err, eul, pos_pid):
        intersection_point = pos - str_err 
        next_waypt = pos - wp_err
        ref_pos = helper.calculate_safe_sliding_bound(next_waypt, intersection_point, distance=23)
        inert_ref_vel = self.pos_controller(ref_pos, pos, pos_pid)
        inert_ref_vel = np.clip(inert_ref_vel, -self.max_vel, self.max_vel)
        inert_ref_vel_leashed = helper.vel_leash(inert_ref_vel, eul, max_vel)
        inert_ref_vel_leashed = inert_ref_vel_leashed

        return np.array([inert_ref_vel_leashed[0], inert_ref_vel_leashed[1], inert_ref_vel[2]])

    def compute_delta_vel(self, lstm_input, rl_input):
        disturbance_pred = self.disturbance_estimator(torch.Tensor(np.array(lstm_input)).unsqueeze(0).to(device)).cpu().detach().numpy()[0]
        
        rl_state = normalize_rl(rl_input)
        action = self.rl_agent.predict(rl_state, deterministic=True)[0]
        return action

    def load_lstm():
        lstm = LSTM(9, 64, 2, 3).to(self.device)
        lstm.load_state_dict(torch.load('../saved_models/lstm_disturbance_4hz.pth'))
        return lstm

    def load_rl():
        agent = PPO.load('../saved_models/rl_agent') 
        return agent 

