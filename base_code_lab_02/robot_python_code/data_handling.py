# External Libraries
import matplotlib.pyplot as plt
from pathlib import Path
import math
import numpy as np

# Internal Libraries
import parameters
import robot_python_code
import motion_models

# Open a file and return data in a form ready to plot
def get_file_data(filename):
    data_loader = robot_python_code.DataLoader(filename)
    data_dict = data_loader.load()

    # The dictionary should have keys ['time', 'control_signal', 'robot_sensor_signal', 'camera_sensor_signal']
    time_list = data_dict['time']
    control_signal_list = data_dict['control_signal']
    robot_sensor_signal_list = data_dict['robot_sensor_signal']
    encoder_count_list = []
    velocity_list = []
    steering_angle_list = []
    for row in robot_sensor_signal_list:
        encoder_count_list.append(row.encoder_counts - robot_sensor_signal_list[0].encoder_counts) # zero the encoder counts to start at 0
    for row in control_signal_list:
        velocity_list.append(row[0])
        steering_angle_list.append(row[1])
    
    return time_list, encoder_count_list, velocity_list, steering_angle_list


# For a given trial, plot the encoder counts, velocities, steering angles
def plot_trial_basics(filename):
    time_list, encoder_count_list, velocity_list, steering_angle_list = get_file_data(filename)
    t0 = time_list[0]
    time_list = [t - t0 for t in time_list]
    plt.plot(time_list, encoder_count_list)
    plt.title('Encoder Values')
    plt.show()
    plt.plot(time_list, velocity_list)
    plt.title('Speed')
    plt.show()
    plt.plot(time_list, steering_angle_list)
    plt.title('Steering')
    plt.show()

xy_measured = {
    'straight': [[36.5, -0.3],
    [36.3, 0.4],
    [35.7, 0.2],
    [77.7, -0.9],
    [79.4, -1.2],
    [80.3, -1.1],
    [130.2, -5.7],
    [130.7, -4.6],
    [130.2, -1.1],],
    'steer3': [
        [23.9, -5.7, -36],#[3, -20, 30, 
        [31.5, 5.8, 24],#[3, 20, 30, 
        [69.9, -21.5, -32],#[3, -10, 45, 
        [80.2, 23, 36],#[3, 10, 45, 
        [102.4, -17.6, -22],#[3, -5, 60, 
        [108.3, 17.1, 22],#[3, 5, 60, 
    ], 
    'steer6' : [
        [46.4, -29.1, -70],#[6, -20, 30,
        [31.5, 32.3, 58],#[6, 20, 30,
        [119, -76.2, -70],#[6, -10, 45,
        [126.8, 91.5, 79],#[6, 10, 45,
        [188.6, -69.9, -39],#[6, -5, 60,
        [198.5, 21.5, 48],#[6, 5, 60,
    ]
}
plot_idx = {'straight': 0, 'steer3': 0, 'steer6': 0}

# Plot a trajectory using the motion model, input data ste from a single trial.
def run_my_model_on_trial(filename, show_plot = True, plot_color = 'ko', category = 'straight'):
    time_list, encoder_count_list, velocity_list, steering_angle_list = get_file_data(filename)
    
    motion_model = motion_models.MyMotionModel([0,0,0], 0)
    x_list, y_list, theta_list = motion_model.traj_propagation(time_list, encoder_count_list, steering_angle_list)
    if category != 'loop_data':
        # plot final position from the trial data
        plt.plot(xy_measured[category][plot_idx[category]][0]*0.01, xy_measured[category][plot_idx[category]][1]*0.01, plot_color.replace('.', 'x'))
        plot_idx[category] += 1
    else:
        plt.plot(x_list[0], y_list[0], 'go')
        plt.plot(x_list[-1], y_list[-1], 'bo')
        plt.legend(["Start", "End"])
    plt.plot(x_list, y_list, plot_color)
    plt.title('Motion Model Predicted XY Traj (m)')
    # plt.axis([-0.5, 1.5, -1, 1])
    if show_plot:
        plt.show()


# Iterate through many trials and plot them as trajectories with motion model
def plot_many_trial_predictions(directory):
    directory_path = Path(directory)
    plot_color_list = ['r.','k.','g.','c.', 'b.', 'r.','k.','g.','c.', 'b.','r.','k.','g.','c.', 'b.', 'r.','k.','g.','c.', 'b.']
    count = 0
    for item in sorted(directory_path.rglob('*.pkl')):
        filename = item.name
        plot_color = plot_color_list[count % len(plot_color_list)]
        print(item)
        run_my_model_on_trial(str(item), False, plot_color, category=item.parent.name)
        count += 1
    plt.show()

# Calculate the predicted distance from single trial for a motion model
def run_my_model_to_predict_distance(filename):
    time_list, encoder_count_list, velocity_list, steering_angle_list = get_file_data(filename)
    motion_model = motion_models.MyMotionModel([0,0,0], 0)
    x_list, _, _ = motion_model.traj_propagation(time_list, encoder_count_list, steering_angle_list)
    distance = x_list[-30]
    
    return distance

# Calculate the differences between two lists, and square them.
def get_diff_squared(m_list,p_list):
    diff_squared_list = []
    for i in range(len(m_list)):
        diff_squared = math.pow(m_list[i]-p_list[i],2)
        diff_squared_list.append(diff_squared)

    coefficients = np.polyfit(m_list, diff_squared_list, 2)
    p=np.poly1d(coefficients)

    plt.plot(m_list, diff_squared_list,'ko')
    plt.plot(m_list, p(m_list),'ro')
    plt.title("Error Squared (m^2)")
    plt.xlabel('Measured distance travelled (m)')
    plt.ylabel('(Actual - Predicted)^2 (m^2)')
    plt.show()

    return diff_squared_list


# Open files, plot them to predict with the motion model, and compare with real values
def process_files_and_plot(files_and_data, directory):
    predicted_distance_list = []
    measured_distance_list = []
    for row in files_and_data:
        filename = row[0]
        measured_distance = row[1]
        measured_distance_list.append(measured_distance)
        predicted_distance = run_my_model_to_predict_distance(directory + filename)
        predicted_distance_list.append(predicted_distance)

    # Plot predicted and measured distance travelled.
    plt.plot(measured_distance_list+[0], predicted_distance_list+[0], 'ko')
    plt.plot([0,1.7],[0,1.7])
    plt.title('Distance Trials')
    plt.xlabel('Measured Distance (m)')
    plt.ylabel('Predicted Distance (m)')
    plt.legend(['Measured vs Predicted', 'Slope 1 Line'])
    plt.show()

    # Plot the associated variance
    get_diff_squared(measured_distance_list, predicted_distance_list)


# Sample and plot some simulated trials
def sample_model(num_samples):
    traj_duration = 10
    for i in range(num_samples):
        model = motion_models.MyMotionModel([0,0,0], 0)
        traj_x, traj_y, traj_theta = model.generate_simulated_traj(traj_duration)
        plt.plot(traj_x, traj_y, 'k.')

    plt.title('Sampling the model')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.show()


######### MAIN ########

# Some sample data to test with
files_and_data = [
    # ['robot_data_60_0_28_01_26_13_41_44.pkl', 67/100], # filename, measured distance in meters
    # ['robot_data_60_0_28_01_26_13_43_41.pkl', 68/100],
    # ['robot_data_60_0_28_01_26_13_37_15.pkl', 113/100],
    # ['robot_data_60_0_28_01_26_13_35_18.pkl', 107/100],
    # ['robot_data_60_0_28_01_26_13_41_10.pkl', 65/100],
    # ['robot_data_60_0_28_01_26_13_42_55.pkl', 70/100],
    # ['robot_data_60_0_28_01_26_13_39_36.pkl', 138/100],
    # ['robot_data_60_0_28_01_26_13_42_19.pkl', 69/100],
    # ['robot_data_60_0_28_01_26_13_36_10.pkl', 109/100],
    # ['robot_data_60_0_28_01_26_13_33_20.pkl', 100/100],
    # ['robot_data_60_0_04_02_26_19_14_15.pkl', 103/100],
    ]

# Plot the motion model predictions for a single trial
if False:
    filename = './data_straight/robot_data_60_0_28_01_26_13_36_10.pkl'
    run_my_model_on_trial(filename)

# Plot the motion model predictions for each trial in a folder
if False:
    directory = ('robot_python_code/data')
    plot_many_trial_predictions(directory)

if True:
    directory = ('robot_python_code/loop_data')
    plot_many_trial_predictions(directory)

# A list of files to open, process, and plot - for comparing predicted with actual distances
if False:
    directory = ('./data/')    
    process_files_and_plot(files_and_data, directory)

# Try to sample with the motion model
if False:
    sample_model(200)
