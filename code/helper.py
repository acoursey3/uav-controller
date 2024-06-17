import numpy as np

def normalize_rl(state):
    state_range = np.zeros(17)
    state_range[0] = 500
    state_range[1] = 500
    state_range[2] = 100 
    state_range[3:6] = 2 * 15
    state_range[6:9] = 2 * np.deg2rad(22.5) * 2
    state_range[9:12] = 2 * np.deg2rad(22.5) * 2
    state_range[12:15] = state_range[:3]
    state_range[15:17] = 100 

    state = state * 2 / (state_range + 1e-6) 
    return state

def normalize_lstm(lstm_input):
    normalization = np.array([1.5, 1.5, 1.5, 15, 15, 15, np.pi/12, np.pi/12, np.pi/12])
    return lstm_input / normalization

def calculate_safe_sliding_bound(reference_point, intersection_point, distance=5):
    # Convert points to numpy arrays for vector calculations
    reference_point = np.array(reference_point)
    intersection_point = np.array(intersection_point)
    
    # Calculate the vector from the point to the reference point
    vector_to_reference = reference_point - intersection_point
    
    # Calculate the distance between the point and the reference point
    distance_to_reference = np.linalg.norm(vector_to_reference)
    
    if distance_to_reference <= distance:
        # If the distance is within the specified range, return the reference point
        return reference_point
    else:
        # Calculate the intermediate point that is 'distance' units along the vector_to_reference
        intermediate_point = intersection_point + (distance / distance_to_reference) * vector_to_reference
        return intermediate_point

def vel_leash(ref_vel, eul, max_vel):
    The = eul[1]
    Phi = eul[0]
    Psi = eul[2]
    R_b_e = np.array([[np.cos(Psi)*np.cos(The), np.cos(Psi)*np.sin(The)*np.sin(Phi)-np.sin(Psi)*np.cos(Phi), np.cos(Psi)*np.sin(The)*np.cos(Phi)+np.sin(Psi)*np.sin(Phi)],
                  [np.sin(Psi)*np.cos(The), np.sin(Psi)*np.sin(The)*np.sin(Phi)+np.cos(Psi)*np.cos(Phi), np.sin(Psi)*np.sin(The)*np.cos(Phi)-np.cos(Psi)*np.sin(Phi)],
                  [-np.sin(The), np.cos(The)*np.sin(Phi), np.cos(The)*np.cos(Phi)]])
    
    velbody = R_b_e.T @ ref_vel
    velbody = velbody[0:2]

    norm = np.linalg.norm(velbody)
    if norm > max_vel:
        velbody = velbody * (max_vel / norm)

    velinert = np.linalg.pinv(R_b_e.T) @ np.array([velbody[0], velbody[1], 0])

    return velinert

def calc_intersection_distance(prev_wp, next_wp, pos):
    current_v = pos - prev_wp
    des_unit_vec = (curr_wp - prev_wp) / np.linalg.norm(curr_wp - prev_wp)

    scalar_factor = np.dot(current_v, des_unit_vec) / (np.dot(des_unit_vec, des_unit_vec)+1e-8)

    intersection_point = prev_wp + scalar_factor * des_unit_vec
    
    intersection_d = -(curr_pos - intersection_point)

    return intersection_d