# External Libraries
import math
import random
import numpy as np

# Motion Model constants


# A function for obtaining variance in distance travelled as a function of distance travelled
def variance_distance_travelled_s(encoder_counts):
    # Add student code here
    a = -5.38659836735178e-08
    b = 0.0002806804940722607 
    c = 0.1051219368000114

    var_s = a * encoder_counts**2 + b * encoder_counts + c

    return var_s * 0.01**2 # convert to meters^2

# Function to calculate distance from encoder counts
def distance_travelled_s(encoder_counts):
    # Add student code here
    s = 0.028308198909415782 * encoder_counts

    return s * 0.01 # convert to meters

# A function for obtaining variance in distance travelled as a function of distance travelled
def variance_rotational_velocity_w(steering_angle_command, speed_command):
    # Add student code here
    c = 0.9126683501683492
    var_w = c

    return var_w

def rotational_velocity_w(steering_angle_command, speed_command):
    # Add student code here
    def model_rot(xdata, a, b, c, d, e):
        return (
            a * xdata[0]
            + b * xdata[1]
            + c * xdata[0] ** 2
            + d * xdata[1] ** 2
            + e * xdata[0] * xdata[1]
        )
    p = [-0.59469697,  0.10597015, -0.00808458, -0.00160586,  0.03744949]
    w = model_rot((steering_angle_command, speed_command), *p)
    
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
        distance = distance_travelled_s(encoder_counts - self.last_encoder_count)
        # distance += \
        #         random.normalvariate(0, math.sqrt(variance_distance_travelled_s(encoder_counts - self.last_encoder_count))) # convert to distance travelled since last step, and add noise based on variance of distance travelled
        self.last_encoder_count = encoder_counts
        # our tested cmd to actual speed ratio is speed / cmd = 0.42
        w = rotational_velocity_w(steering_angle_command, distance / delta_t / 0.42 * 100)
        # w += np.sign(w) * \
        # random.normalvariate(0, math.sqrt(max(0, variance_rotational_velocity_w(steering_angle_command, distance / delta_t / 0.42)))) # convert to equivalent speed command for the model, and add noise based on variance of rotational velocity
        self.state[2] = self.state[2] + w / 180 * math.pi * delta_t
        self.state[0] = self.state[0] + distance * math.cos(self.state[2])
        self.state[1] = self.state[1] + distance * math.sin(self.state[2])
        
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
            