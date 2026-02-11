# External Libraries
from json import encoder
import math
import random
# Motion Model constants

TICKS_PER_METER = 3481.84
METERS_PER_TICK = 1.0 / TICKS_PER_METER
LOGISTIC_A_DEG_PER_S = -11.552156
LOGISTIC_B = 0.103852
VARIANCE_DISTANCE_M2 = 6.23263e-05
VARIANCE_ANGULAR_VELOCITY_DEG2_PER_S2 = 0.687878

# A function for obtaining variance in distance travelled as a function of distance travelled
def variance_distance_travelled_s(distance):
    # Mean variance from encoder_to_distance.py calibration (m^2), constant model.
    _ = distance
    var_s = VARIANCE_DISTANCE_M2

    return var_s

# Function to calculate distance from encoder counts
def distance_travelled_s(encoder_counts):
    # Add student code here
    s = METERS_PER_TICK * float(encoder_counts)
    
    return s

# A function for obtaining variance in distance travelled as a function of distance travelled
def variance_rotational_velocity_w(distance):
    # Mean variance from steer_to_yaw.py calibration ((deg/s)^2), constant model.
    _ = distance
    var_w = VARIANCE_ANGULAR_VELOCITY_DEG2_PER_S2

    return var_w

def rotational_velocity_w(steering_angle_command):
    # Calibrated from steer_to_yaw.py:
    # omega_deg_per_s = a * (2 / (1 + exp(-b*steering)) - 1)
    steering = float(steering_angle_command)
    omega_deg_per_s = LOGISTIC_A_DEG_PER_S * (2.0 / (1.0 + math.exp(-LOGISTIC_B * steering)) - 1.0)
    w = math.radians(omega_deg_per_s)
    
    return w

# This class is an example structure for implementing your motion model.
class MyMotionModel:

    # Constructor, change as you see fit.
    def __init__(self, initial_state, last_encoder_count):
        self.state = initial_state
        self.last_encoder_count = last_encoder_count

    # This is the key step of your motion model, which implements x_t = f(x_{t-1}, u_t)
    def step_update(self, encoder_counts, steering_angle_command, delta_t):
        # Add student code here
        old_state = self.state
        Delta_s = distance_travelled_s(encoder_counts - self.last_encoder_count)
        omega = rotational_velocity_w(steering_angle_command)
        theta = old_state[2]
        new_state = old_state

        #x
        new_state[0] += Delta_s * math.cos(theta)
        #y
        new_state[1] += Delta_s  * math.sin(theta)
        #theta
        new_state[2] += omega * delta_t

        self.last_encoder_count = encoder_counts

        return self.state
    
    # This is a great tool to take in data from a trial and iterate over the data to create 
    # a robot trajectory in the global frame, using your motion model.
    def traj_propagation(self, time_list, encoder_count_list, steering_angle_list):
        x_list = [self.state[0]]
        y_list = [self.state[1]]
        theta_list = [self.state[2]]
        self.last_encoder_count = encoder_count_list[0]
        for i in range(1, len(encoder_count_list)):
            delta_t = time_list[i] - time_list[i-1]
            new_state = self.step_update(encoder_count_list[i], steering_angle_list[i], delta_t)
            x_list.append(new_state[0])
            y_list.append(new_state[1])
            theta_list.append(new_state[2])

        return x_list, y_list, theta_list
    

    # Coming soon
    def generate_simulated_traj(self, duration):
        delta_t = 0.1
        t_list = []
        x_list = []
        y_list = []
        theta_list = []
        t = 0
        encoder_counts = 0
        while t < duration:

            t += delta_t 
        return t_list, x_list, y_list, theta_list
            
