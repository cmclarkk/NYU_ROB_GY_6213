from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# filename, yaw change in degrees (signed)
files_and_data = [
    ["robot_data_50_-5_06_02_26_16_19_28.pkl", 18.0],
    ["robot_data_50_-10_06_02_26_16_19_49.pkl", 30.0],
    ["robot_data_50_-20_06_02_26_16_20_04.pkl", 42.0],
    ["robot_data_50_5_06_02_26_16_18_48.pkl", -9.0],
    ["robot_data_50_10_06_02_26_16_18_04.pkl", -24.0],
    ["robot_data_50_20_06_02_26_16_16_28.pkl", -48.0],
    #new data
    ["robot_data_50_0_06_02_26_16_00_04.pkl", 2.0],
    ["robot_data_50_0_06_02_26_16_02_19.pkl", 0.0],
    ["robot_data_50_0_06_02_26_16_03_13.pkl", -2.0],
    ["robot_data__50_-20_10_02_26_22_39_15.pkl", 32.0],
    ["robot_data__50_-20_10_02_26_22_41_07.pkl", 30.0],
    ["robot_data__50_-20_10_02_26_22_42_49.pkl", 32.0],
    ["robot_data__50_-15_10_02_26_22_34_38.pkl", 30.0],
    ["robot_data__50_-15_10_02_26_22_35_51.pkl", 32.0],
    ["robot_data__50_-15_10_02_26_22_37_01.pkl", 32.0],
    ["robot_data__50_-10_10_02_26_22_25_45.pkl", 25.0],
    ["robot_data__50_-10_10_02_26_22_30_50.pkl", 26.0],
    ["robot_data__50_-10_10_02_26_22_33_17.pkl", 26.0],
    ["robot_data__50_-5_10_02_26_22_15_39.pkl", 14.0],
    ["robot_data__50_-5_10_02_26_22_21_13.pkl", 13.0],
    ["robot_data__50_-5_10_02_26_22_23_06.pkl", 15.0],
    ["robot_data__50_5_10_02_26_21_51_12.pkl", -4.0],
    ["robot_data__50_5_10_02_26_22_44_26.pkl", -5.0],
    ["robot_data__50_5_10_02_26_22_46_27.pkl", -5.0],
    ["robot_data__50_10_10_02_26_21_53_44.pkl", -21.0],
    ["robot_data__50_10_10_02_26_21_55_31.pkl", -19.0],
    ["robot_data__50_10_10_02_26_21_57_15.pkl", -20.0],
    ["robot_data__50_15_10_02_26_22_00_12.pkl", -29.0],
    ["robot_data__50_15_10_02_26_22_05_05.pkl", -30.0],
    ["robot_data__50_15_10_02_26_22_07_37.pkl", -31.0],
    ["robot_data__50_20_10_02_26_22_09_20.pkl", -38.0],
    ["robot_data__50_20_10_02_26_22_10_41.pkl", -38.0],
    ["robot_data__50_20_10_02_26_22_12_35.pkl", -37.0],
]


DATA_DIRECTORY = Path(__file__).resolve().parent / "data_curved_new"


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


def build_trial_result(data_directory: Path, filename: str, yaw_change_deg: float) -> TrialResult:
    data_dict = load_data_file(data_directory / filename)
    time_list = data_dict["time"]
    control_signal_list = data_dict["control_signal"]
    steering_control, time_taken = get_time_taken_for_steering(time_list, control_signal_list)

    angular_velocity = yaw_change_deg / time_taken
    return TrialResult(filename, steering_control, yaw_change_deg, time_taken, angular_velocity)


def logistic_centered(
    steering_control_list: np.ndarray,
    logistic_a: float,
    logistic_b: float,
) -> np.ndarray:
    return logistic_a * (2.0 / (1.0 + np.exp(-logistic_b * steering_control_list)) - 1.0)


def fit_logistic_centered(
    steering_control_list: np.ndarray,
    angular_velocity_list: np.ndarray,
) -> tuple[float, float]:
    if int(np.unique(steering_control_list).size) < 2:
        raise ValueError("Need at least two distinct steering inputs to fit a logistic model.")

    initial_a = float(np.max(np.abs(angular_velocity_list)))
    if initial_a == 0.0:
        initial_a = 1.0

    linear_slope = float(np.polyfit(steering_control_list, angular_velocity_list, deg=1)[0])
    initial_b = max(1e-3, min(10.0, abs(2.0 * linear_slope / initial_a)))

    parameters, _ = curve_fit(
        logistic_centered,
        steering_control_list,
        angular_velocity_list,
        p0=(initial_a, initial_b),
        bounds=([-np.inf, 1e-6], [np.inf, np.inf]),
        maxfev=20000,
    )
    logistic_a, logistic_b = parameters
    return float(logistic_a), float(logistic_b)


def detect_outlier_indices(
    values: np.ndarray,
    max_outliers: int = 3,
) -> list[int]:
    """
    Detect the strongest `max_outliers` points using iterative MAD scoring.
    Returns original indices for detected outliers.
    """

    if values.size < 3 or max_outliers <= 0:
        return []

    remaining_indices = list(range(values.size))
    detected: list[int] = []
    target_count = min(max_outliers, max(0, values.size - 1))

    while len(remaining_indices) >= 2 and len(detected) < target_count:
        remaining_values = values[remaining_indices]
        median_value = float(np.median(remaining_values))
        abs_deviations = np.abs(remaining_values - median_value)
        mad = float(np.median(abs_deviations))
        if mad == 0.0:
            local_index = int(np.argmax(np.abs(remaining_values - median_value)))
        else:
            modified_z_scores = 0.6745 * (remaining_values - median_value) / mad
            local_index = int(np.argmax(np.abs(modified_z_scores)))

        detected.append(remaining_indices[local_index])
        remaining_indices.pop(local_index)

    return sorted(detected)


def plot_steering_velocity_and_variance(
    steering_control_list: np.ndarray,
    angular_velocity_list: np.ndarray,
    logistic_a: float,
    logistic_b: float,
    variance_list: np.ndarray,
    outlier_indices: list[int],
    variance_mean: float,
) -> None:
    x_plot = np.linspace(float(np.min(steering_control_list)), float(np.max(steering_control_list)), 200)
    velocity_fit_plot = logistic_centered(x_plot, logistic_a, logistic_b)
    predicted_angular_velocity = logistic_centered(steering_control_list, logistic_a, logistic_b)
    rmse = float(np.sqrt(np.mean(np.square(angular_velocity_list - predicted_angular_velocity))))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=False)

    # Top subplot: steering control vs angular velocity.
    ax1.plot(steering_control_list, angular_velocity_list, "ko")
    ax1.plot(x_plot, velocity_fit_plot, "r-")
    ax1.set_xlabel("Steering control")
    ax1.set_ylabel("Angular velocity (deg/s)")
    ax1.set_title("Steering Control vs Angular Velocity")
    equation_label = rf"y = ({logistic_a:.4g})(2/(1+e^(-({logistic_b:.4g})x)) - 1), RMSE = {rmse:.4g} deg/s"
    ax1.legend(["Data", equation_label], loc="upper center", bbox_to_anchor=(0.5, -0.18))
    ax1.axhline(0.0, color="0.4", linewidth=1.0)
    ax1.axvline(0.0, color="0.4", linewidth=1.0)
    ax1.set_box_aspect(0.5)
    ax1.grid(True)

    # Bottom subplot: variance with mean-fit line and red outliers.
    if not outlier_indices:
        variance_data_handle, = ax2.plot(steering_control_list, variance_list, "ko")
        outlier_handle = None
    else:
        inlier_mask = np.ones(variance_list.shape[0], dtype=bool)
        inlier_mask[outlier_indices] = False
        variance_data_handle, = ax2.plot(steering_control_list[inlier_mask], variance_list[inlier_mask], "ko")
        outlier_handle, = ax2.plot(
            steering_control_list[outlier_indices],
            variance_list[outlier_indices],
            "ro",
            markersize=7,
        )
    variance_mean_handle = ax2.axhline(variance_mean, color="b", linewidth=1.5)
    ax2.set_xlabel("Steering control")
    ax2.set_ylabel(r"Variance $(\omega - \hat{\omega})^2$ ((deg/s)$^2$)")
    ax2.set_title("Variance vs Steering Control")
    ax2.set_ylim(bottom=0.0)
    mean_label = rf"Mean variance (inliers) = {variance_mean:.4g} (deg/s)$^2$"
    if outlier_handle is None:
        ax2.legend(
            [variance_data_handle, variance_mean_handle],
            ["Data", mean_label],
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
        )
    else:
        ax2.legend(
            [variance_data_handle, outlier_handle, variance_mean_handle],
            ["Inliers", "Outliers", mean_label],
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
        )
    ax2.axhline(0.0, color="0.4", linewidth=1.0)
    ax2.axvline(0.0, color="0.4", linewidth=1.0)
    ax2.set_box_aspect(0.5)
    ax2.grid(True)

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.55, bottom=0.12)
    plt.show()


def main(show_plot: bool = True, show_variance_plot: bool = True) -> None:
    trial_results = [build_trial_result(DATA_DIRECTORY, filename, yaw_change) for filename, yaw_change in files_and_data]

    steering_control_list = np.array([trial.steering_control for trial in trial_results], dtype=float)
    angular_velocity_list = np.array([trial.angular_velocity_deg_per_s for trial in trial_results], dtype=float)
    logistic_a, logistic_b = fit_logistic_centered(steering_control_list, angular_velocity_list)
    predicted_angular_velocity = logistic_centered(steering_control_list, logistic_a, logistic_b)
    variance_list = np.square(angular_velocity_list - predicted_angular_velocity)
    outlier_indices = detect_outlier_indices(variance_list, max_outliers=0)
    inlier_mask = np.ones(variance_list.shape[0], dtype=bool)
    inlier_mask[outlier_indices] = False
    inlier_variance = variance_list[inlier_mask]
    variance_mean = float(np.mean(inlier_variance))

    # print("Per-trial results:")
    # for trial in trial_results:
    #     print(
    #         f"{trial.filename}: steering={trial.steering_control:.1f}, "
    #         f"time={trial.time_taken_s:.3f} s, yaw={trial.yaw_change_deg:.2f} deg, "
    #         f"omega={trial.angular_velocity_deg_per_s:.4f} deg/s"
    #     )
    print("\nCentered logistic mapping: angular_velocity = a * (2/(1 + exp(-b*steering_control)) - 1)")
    print(f"a = {logistic_a:.6f}, b = {logistic_b:.6f}")
    print("Variance model (inlier mean):")
    print(f"variance = {variance_mean:.6f} (deg/s)^2")
    for outlier_index in outlier_indices:
        print(
            f"Excluded variance outlier: steering={steering_control_list[outlier_index]:.1f}, "
            f"variance={variance_list[outlier_index]:.6f}"
        )

    if show_plot or show_variance_plot:
        plot_steering_velocity_and_variance(
            steering_control_list,
            angular_velocity_list,
            logistic_a,
            logistic_b,
            variance_list,
            outlier_indices,
            variance_mean,
        )


if __name__ == "__main__":
    main(show_plot=True, show_variance_plot=True)
