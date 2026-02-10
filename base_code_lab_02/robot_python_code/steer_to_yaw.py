from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


# filename, yaw change magnitude in degrees
files_and_data: list[tuple[str, float]] = [
    ("robot_data_50_-5_06_02_26_16_19_28.pkl", 18.0),
    ("robot_data_50_-10_06_02_26_16_19_49.pkl", 30.0),
    ("robot_data_50_-20_06_02_26_16_20_04.pkl", 42.0),
    ("robot_data_50_5_06_02_26_16_18_48.pkl", 9.0),
    ("robot_data_50_10_06_02_26_16_18_04.pkl", 24.0),
    ("robot_data_50_20_06_02_26_16_16_28.pkl", 48.0),
]


DATA_DIRECTORY = Path(__file__).resolve().parent / "data_curved"


class _RobotSensorSignal:
    pass


class _RobotDataUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if module == "robot_python_code" and name == "RobotSensorSignal":
            return _RobotSensorSignal
        return super().find_class(module, name)


@dataclass
class TrialResult:
    filename: str
    steering_control: float
    yaw_change_deg: float
    time_taken_s: float
    angular_velocity_deg_per_s: float


def load_data_file(filename: Path) -> dict:
    with filename.open("rb") as file_handle:
        data_dict = _RobotDataUnpickler(file_handle).load()
    return data_dict


def get_time_taken_for_steering(time_list: list[float], control_signal_list: list[list[float]]) -> tuple[float, float]:
    steering_control = 0.0
    for row in control_signal_list:
        if row[1] != 0:
            steering_control = float(row[1])
            break

    active_indices = [i for i, row in enumerate(control_signal_list) if row[1] == steering_control and steering_control != 0]
    if len(active_indices) >= 2:
        time_taken = float(time_list[active_indices[-1]] - time_list[active_indices[0]])
    else:
        time_taken = float(time_list[-1] - time_list[0])

    return steering_control, time_taken


def build_trial_result(data_directory: Path, filename: str, yaw_change_deg_magnitude: float) -> TrialResult:
    data_dict = load_data_file(data_directory / filename)
    time_list = data_dict["time"]
    control_signal_list = data_dict["control_signal"]
    steering_control, time_taken = get_time_taken_for_steering(time_list, control_signal_list)

    yaw_change_signed_deg = float(np.copysign(yaw_change_deg_magnitude, steering_control))
    angular_velocity = yaw_change_signed_deg / time_taken
    return TrialResult(filename, steering_control, yaw_change_signed_deg, time_taken, angular_velocity)


def fit_y_equals_ax(steering_control_list: np.ndarray, angular_velocity_list: np.ndarray) -> float:
    denominator = float(np.dot(steering_control_list, steering_control_list))
    if denominator == 0.0:
        raise ValueError("Cannot fit y = ax when all steering inputs are zero.")
    return float(np.dot(steering_control_list, angular_velocity_list) / denominator)


def plot_steering_vs_angular_velocity(steering_control_list: np.ndarray, angular_velocity_list: np.ndarray, slope_a: float) -> None:
    x_plot = np.linspace(float(np.min(steering_control_list)) * 1.1, float(np.max(steering_control_list)) * 1.1, 200)
    y_plot = slope_a * x_plot

    plt.figure()
    plt.plot(steering_control_list, angular_velocity_list, "ko")
    plt.plot(x_plot, y_plot, "r-")
    plt.xlabel("Steering control")
    plt.ylabel("Angular velocity (deg/s)")
    plt.title("Steering Control vs Angular Velocity")
    plt.legend(["Data", "Linear fit: y = ax"])
    plt.grid(True)
    plt.show()


def main(show_plot: bool = True) -> None:
    trial_results = [build_trial_result(DATA_DIRECTORY, filename, yaw_change) for filename, yaw_change in files_and_data]

    steering_control_list = np.array([trial.steering_control for trial in trial_results], dtype=float)
    angular_velocity_list = np.array([trial.angular_velocity_deg_per_s for trial in trial_results], dtype=float)
    slope_a = fit_y_equals_ax(steering_control_list, angular_velocity_list)

    print("Per-trial results:")
    for trial in trial_results:
        print(
            f"{trial.filename}: steering={trial.steering_control:.1f}, "
            f"time={trial.time_taken_s:.3f} s, yaw={trial.yaw_change_deg:.2f} deg, "
            f"omega={trial.angular_velocity_deg_per_s:.4f} deg/s"
        )
    print(f"\nLinear mapping (through origin): angular_velocity = a * steering_control")
    print(f"a = {slope_a:.6f} (deg/s per steering unit)")

    if show_plot:
        plot_steering_vs_angular_velocity(steering_control_list, angular_velocity_list, slope_a)


if __name__ == "__main__":
    main(show_plot=True)
