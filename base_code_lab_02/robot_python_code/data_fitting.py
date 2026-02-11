# External Libraries
import matplotlib.pyplot as plt
from pathlib import Path
import math
import numpy as np

# Internal Libraries
import parameters
import robot_python_code
from scipy.optimize import curve_fit
from functools import partial


# Open a file and return data in a form ready to plot
def get_file_data(filename):
    data_loader = robot_python_code.DataLoader(filename)
    data_dict = data_loader.load()

    # The dictionary should have keys ['time', 'control_signal', 'robot_sensor_signal', 'camera_sensor_signal']
    time_list = data_dict["time"]
    control_signal_list = data_dict["control_signal"]
    robot_sensor_signal_list = data_dict["robot_sensor_signal"]
    encoder_count_list = []
    velocity_list = []
    steering_angle_list = []
    for row in robot_sensor_signal_list:
        encoder_count_list.append(row.encoder_counts)
    for row in control_signal_list:
        velocity_list.append(row[0])
        steering_angle_list.append(row[1])

    return time_list, encoder_count_list, velocity_list, steering_angle_list


mode = "fit_linear"


# def fit_data(x_data, y_data, mode='fit_linear'):
def load_file(filename):
    time_list, encoder_count_list, velocity_list, steering_angle_list = get_file_data(
        filename
    )
    t0 = time_list[0]
    time_list = [t - t0 for t in time_list]
    return {
        "filename": filename,
        "t": np.array(time_list),
        "encoder_counts": np.array(encoder_count_list),
        "velocities": np.array(velocity_list),
        "steering_angles": np.array(steering_angle_list),
    }


def load_files(file_list):
    all_data = []
    for filename in file_list:
        data = load_file(filename)
        all_data.append(data)
    # merge data by keys into single list [array, array, ...]
    merged_data = {}
    for key in all_data[0].keys():
        merged_data[key] = [d[key] for d in all_data]
    return merged_data


def linear_no_intercept(x, a):
    return a * x


def linear_with_intercept(x, a, b):
    return a * x + b


def quad_no_intercept(x, a, b, c):
    return a * x**2 + b * x + c


def fit_linear(encoder_counts, distance):
    """Fit a linear model to the encoder counts and distance measured. distance = f(encoder_counts)"""
    encoder_counts = np.array([0.0] + [e[-1] - e[0] for e in encoder_counts])
    distance = np.linalg.norm(np.array([[0, 0]] + distance), axis=1)
    # Calculate the slope and intercept of the linear model
    popt, pcov = curve_fit(linear_no_intercept, encoder_counts, distance)
    slope = popt[0]
    intercept = 0  # No intercept in this model
    # Print the slope and intercept
    print(f"f_se fitting param: Slope: {slope}, Intercept: {intercept}")
    # Calculate the fitted values
    fitted_values = slope * encoder_counts + intercept
    # compute the prediction squared error s^2 and fit s^2 = f(encoder_counts)
    residuals = distance - fitted_values
    squared_errors = residuals**2
    popt_error, _ = curve_fit(quad_no_intercept, encoder_counts, squared_errors)
    lpopt_error, _ = curve_fit(linear_with_intercept, encoder_counts, squared_errors)
    print(
        "quadratic fit parameters for error: a =",
        popt_error[0],
        "b =",
        popt_error[1],
        "c =",
        popt_error[2],
    )
    print("linear fit parameters for error: a =", lpopt_error[0])
    # Plot the data and the fitted line
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3))

    axes[0].scatter(encoder_counts, distance, label="Measured Data", color="blue")
    axes[0].plot(encoder_counts, fitted_values, label="Fitted Line", color="red")
    axes[0].set_xlabel("Encoder Counts")

    axes[0].set_ylabel("Distance Measured (cm)")
    axes[0].set_title("f_se")
    axes[0].legend()
    axes[0].grid()

    axes[1].scatter(
        encoder_counts, squared_errors, label="Squared Error", color="purple"
    )
    error_plot_interpolation = np.linspace(
        min(encoder_counts), max(encoder_counts), 100
    )
    axes[1].plot(
        error_plot_interpolation,
        np.vstack(
            [
                quad_no_intercept(error_plot_interpolation, *popt_error),
                linear_with_intercept(error_plot_interpolation, *lpopt_error),
            ]
        ).T,
        label=["Quadratic Fit", "Linear Fit"],
    )
    axes[1].set_xlabel("Encoder Counts")
    axes[1].set_ylabel("Squared Error (cm^2)")
    axes[1].set_title("f_{ss}")
    axes[1].legend()
    axes[1].grid()

    plt.tight_layout()
    plt.show()


def load_linear():
    # load all files in robot_python_code/data/straight, sort by name
    data_path = Path("robot_python_code/data/straight")
    all_files = sorted(data_path.glob("*.pkl"))
    print(f"Loading files: {all_files}")
    all_data = load_files(all_files)
    # fit linear model
    encoder_counts = all_data["encoder_counts"]
    distance_measured = [
        [36.5, -0.3],
        [36.3, 0.4],
        [35.7, 0.2],
        [77.7, -0.9],
        [79.4, -1.2],
        [80.3, -1.1],
        [130.2, -5.7],
        [130.7, -4.6],
        [130.2, -1.1],
    ]
    fit_linear(encoder_counts, distance_measured)


def fit_rot(yaw_rate, steering_angles, velocities):
    """Fit a model yaw_rate = f(steering_angles, velocities) to the data. We want to find a function that can predict the final orientation of the robot given the steering angles and velocities."""

    # For simplicity, we will fit a linear model: yaw_rate = a * steering_angle + b * velocity + c
    # We can use curve_fit to find the parameters a, b, c
    def model(xdata, a, b, c):
        return np.ones_like(xdata[0])*c

    def model_rot(xdata, a, b, c, d, e, f, with_intercept=True):
        return (
            a * xdata[0]
            # + b * xdata[1]
            + c * xdata[0] ** 3 * xdata[1]
            # + d * xdata[0] ** 2 * xdata[1]
            + e * xdata[0] * xdata[1]
            + (f if with_intercept else 0)
        )

    value_model = partial(model_rot, with_intercept=True)
    popt, pcov = curve_fit(
        value_model,
        (np.concatenate((steering_angles.flatten(), np.zeros(1))), 
         np.concatenate((velocities.flatten(), np.zeros(1)))),
        np.concatenate((yaw_rate.flatten(), np.zeros(1))),
    )
    print("rot Fitted parameters:", popt)
    print("eval at steering 10, velocity 5:", value_model((10, 5), *popt))

    # Calculate fitted values and squared errors
    fitted_values = value_model(
        (steering_angles.flatten(), velocities.flatten()), *popt
    )
    residuals = yaw_rate.flatten() - fitted_values
    squared_errors = residuals**2

    error_model = model
    popt_error_quad, _ = curve_fit(
        error_model,
        (steering_angles.flatten(), velocities.flatten()),
        squared_errors.flatten(),
    )
    print(
        "Error model parameters: a =",
        popt_error_quad[0],
        "b =",
        popt_error_quad[1],
        "c =",
        popt_error_quad[2],
    )

    # Sort by angle first, then velocity for plotting
    sorted_indices = np.lexsort((velocities.flatten(), steering_angles.flatten()))
    sorted_steering = steering_angles.flatten()[sorted_indices]
    sorted_velocities = velocities.flatten()[sorted_indices]
    sorted_yaw_rate = yaw_rate.flatten()[sorted_indices]
    sorted_fitted = fitted_values.flatten()[sorted_indices]
    sorted_errors = squared_errors.flatten()[sorted_indices]

    # Plot the data and the fitted model
    fig = plt.figure(figsize=(12, 8))

    # Original fit - yaw_rate vs steering angle
    ax1 = fig.add_subplot(221)
    ax1.scatter(sorted_steering, sorted_yaw_rate, label="Measured Data", color="blue")

    # Create smooth interpolation for plotting
    smooth_steering = np.linspace(sorted_steering.min(), sorted_steering.max(), 100)
    # Use the corresponding velocity value for each steering angle
    smooth_velocities = np.interp(smooth_steering, sorted_steering, sorted_velocities)
    smooth_fitted = value_model((smooth_steering, smooth_velocities), *popt)

    ax1.plot(smooth_steering, smooth_fitted, label="Fitted Line", color="red")
    ax1.set_xlabel("Steering Angle (degrees)")
    ax1.set_ylabel("Yaw Rate (degrees/s)")
    ax1.set_title("θ vs Steering Angle")
    ax1.legend()
    ax1.grid()

    # Original fit - yaw_rate vs velocity, split by angle sign
    ax3 = fig.add_subplot(223)
    positive_angle_mask = sorted_steering >= 0
    negative_angle_mask = sorted_steering < 0

    # Create smooth interpolation for plotting
    smooth_velocities = np.linspace(
        sorted_velocities.min(), sorted_velocities.max(), 100
    )
    smooth_sorted_steering_negative = np.interp(
        smooth_velocities,
        sorted_velocities[negative_angle_mask],
        sorted_steering[negative_angle_mask],
    )
    smooth_sorted_steering_positive = -smooth_sorted_steering_negative
    ax3.scatter(
        sorted_velocities[positive_angle_mask],
        sorted_yaw_rate[positive_angle_mask],
        label="Positive Angle",
        color="blue",
    )
    ax3.scatter(
        sorted_velocities[negative_angle_mask],
        sorted_yaw_rate[negative_angle_mask],
        label="Negative Angle",
        color="orange",
    )
    ax3.plot(
        smooth_velocities,
        value_model((smooth_sorted_steering_positive, smooth_velocities), *popt),
        color="blue",
        linestyle="--",
    )
    ax3.plot(
        smooth_velocities,
        value_model((smooth_sorted_steering_negative, smooth_velocities), *popt),
        color="orange",
        linestyle="--",
    )
    ax3.set_xlabel("Velocity (cm/s)")
    ax3.set_ylabel("Final Orientation (degrees)")
    ax3.set_title("θ vs Velocity")
    ax3.legend()
    ax3.grid()

    # 3D error plot - squared error vs steering angle and velocity
    ax2 = fig.add_subplot(122, projection="3d")
    ax2.scatter(
        sorted_steering,
        sorted_velocities,
        sorted_errors,
        c="purple",
        marker="o",
        label="Squared Error",
    )

    # Create mesh for fitted surface
    steering_mesh = np.linspace(sorted_steering.min(), sorted_steering.max(), 30)
    velocity_mesh = np.linspace(sorted_velocities.min(), sorted_velocities.max(), 30)
    steering_grid, velocity_grid = np.meshgrid(steering_mesh, velocity_mesh)
    error_surface = error_model((steering_grid, velocity_grid), *popt_error_quad)

    ax2.plot_surface(
        steering_grid, velocity_grid, error_surface, alpha=0.5, cmap="viridis"
    )
    ax2.set_xlabel("Steering Angle (degrees)")
    ax2.set_ylabel("Velocity (cm/s)")
    ax2.set_zlabel("Squared Error ((degrees/s)²)")
    ax2.set_title("Error Surface")
    ax2.legend()

    plt.tight_layout()
    plt.show()


def load_rot():
    # duration, steering angle, velocity, x, y, yaw_rate
    measurements = [
        [3, -20, 30, 23.9, -5.7, -36],
        [6, -20, 30, 46.4, -29.1, -70],
        [3, -10, 45, 69.9, -21.5, -32],
        [6, -10, 45, 119, -76.2, -70],
        [3, -5, 60, 102.4, -17.6, -22],
        [6, -5, 60, 188.6, -69.9, -39],
        [3, 5, 60, 108.3, 17.1, 22],
        [6, 5, 60, 198.5, 21.5, 48],
        [3, 10, 45, 80.2, 23, 36],
        [6, 10, 45, 126.8, 91.5, 79],
        [3, 20, 30, 31.5, 5.8, 24],
        [6, 20, 30, 31.5, 32.3, 58],
    ]

    duration3s_path = Path("robot_python_code/data/steer3")
    duration6s_path = Path("robot_python_code/data/steer6")
    all_files_3s = sorted(duration3s_path.glob("*.pkl"))
    all_files_6s = sorted(duration6s_path.glob("*.pkl"))
    print(f"Loading files: {all_files_3s}")
    print(f"Loading files: {all_files_6s}")
    all_data_3s = load_files(all_files_3s)
    all_data_6s = load_files(all_files_6s)
    # match filename in all_data_xx to measurement, e.g. robot_data_speed_angle_...
    # and then merge two datasets
    merged_data = {}
    for duration in [3, 6]:
        data_key = f"all_data_{duration}s"
        data = eval(data_key)
        data["yaw_rate"] = np.zeros(len(data["encoder_counts"]))
        for i, filename in enumerate(data["filename"]):
            # parse angle and velocity from filename
            velocity, angle = map(int, filename.stem.split("_")[2:4])
            # find matching measurement
            for m in measurements:
                if m[0] == duration and m[1] == angle and m[2] == velocity:
                    # add yaw_rate to all_data_3s
                    data["yaw_rate"][i] = m[5] / duration
                    data["steering_angles"][i] = data["steering_angles"][i][0]
                    data["velocities"][i] = data["velocities"][i][0]
        for key in data.keys():
            if key not in merged_data:
                merged_data[key] = []
            merged_data[key].extend(data[key])

    # fit model
    fit_rot(
        np.array(merged_data["yaw_rate"]),
        np.array(merged_data["steering_angles"]),
        np.array(merged_data["velocities"]),
    )


load_linear()
# load_rot()
