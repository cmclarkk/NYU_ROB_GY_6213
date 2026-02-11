from __future__ import annotations

import pickle
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# (filename, measured distance in meters)
files_and_data = [
    ['robot_data_50_0_06_02_26_16_00_04.pkl', 59/100], #5
    ['robot_data_50_0_06_02_26_16_03_13.pkl', 73.5/100], #6
    ['robot_data_50_0_06_02_26_16_04_33.pkl', 79/100], #6.5
    ['robot_data_50_0_06_02_26_16_05_03.pkl', 77/100], #7
    ['robot_data_50_0_06_02_26_16_05_47.pkl', 43.5/100], #4
    ['robot_data_50_0_06_02_26_16_07_32.pkl', 35.5/100], #3.5
    ['robot_data_50_0_06_02_26_16_08_18.pkl', 32/100], #3
    ['robot_data_50_0_06_02_26_16_09_07.pkl', 23/100], #2.5
    ['robot_data_50_0_06_02_26_16_09_40.pkl', 17.5/100], #2
    ['robot_data_50_0_06_02_26_16_10_26.pkl', 89/100], #7.5
    # ['robot_data_50_0_06_02_26_16_10_58.pkl', 95/100], #8
    ]


@dataclass
class RobotSensorSignal:
    """
    Minimal stand-in used for unpickling.

    The log files store instances of `robot_python_code.RobotSensorSignal`, but
    importing that module pulls in optional dependencies (serial/cv2). For this
    script we only need the `encoder_counts` attribute.
    """

    encoder_counts: int = 0


class _RobotLogUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:  # noqa: ANN401
        if name == "RobotSensorSignal" and module.endswith("robot_python_code"):
            return RobotSensorSignal
        return super().find_class(module, name)


def load_log_dict(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return _RobotLogUnpickler(f).load()


def encoder_delta_ticks(data_dict: dict[str, Any], cutoff_from_end: int = 30) -> int:
    """
    Returns encoder ticks traveled during the trial.

    We drop the last `cutoff_from_end` samples to avoid the extra logging tail
    after the robot stops (mirrors the `-30` approach used in `data_handling.py`).
    """

    robot_sensor_signal_list = data_dict["robot_sensor_signal"]
    encoder_counts = [row.encoder_counts for row in robot_sensor_signal_list]
    if not encoder_counts:
        return 0

    start = int(encoder_counts[0])
    if len(encoder_counts) > cutoff_from_end:
        end = int(encoder_counts[-cutoff_from_end])
    else:
        end = int(encoder_counts[-1])

    return abs(end - start)


def fit_slope_m_per_tick(ticks: list[float], distances_m: list[float]) -> float:
    """
    Fit through origin: distance_m ≈ slope * ticks.
    Returns slope_m_per_tick.
    """

    if len(ticks) != len(distances_m):
        raise ValueError("ticks and distances_m must have the same length")
    if len(ticks) < 2:
        raise ValueError("need at least 2 points to fit a line")

    denom = sum(x * x for x in ticks)
    if denom == 0:
        raise ValueError("cannot fit slope: all tick values are identical")

    numer = sum(ticks[i] * distances_m[i] for i in range(len(ticks)))
    slope = numer / denom
    return float(slope)


def detect_outlier_indices(
    values: list[float],
    max_outliers: int = 3,
) -> list[int]:
    """
    Detect the strongest `max_outliers` points using iterative MAD scoring.
    Returns original indices for detected outliers.
    """

    if len(values) < 3 or max_outliers <= 0:
        return []

    remaining_indices = list(range(len(values)))
    detected: list[int] = []
    target_count = min(max_outliers, max(0, len(values) - 1))

    while len(remaining_indices) >= 2 and len(detected) < target_count:
        remaining_values = [values[i] for i in remaining_indices]
        median_value = statistics.median(remaining_values)
        abs_deviations = [abs(v - median_value) for v in remaining_values]
        mad = statistics.median(abs_deviations)
        if mad == 0:
            local_index = max(range(len(remaining_values)), key=lambda i: abs(remaining_values[i] - median_value))
        else:
            modified_z_scores = [0.6745 * (v - median_value) / mad for v in remaining_values]
            local_index = max(range(len(remaining_values)), key=lambda i: abs(modified_z_scores[i]))

        detected.append(remaining_indices[local_index])
        remaining_indices.pop(local_index)

    return sorted(detected)


def main() -> None:
    data_dir = Path(__file__).resolve().parent / "data_straight"

    measured_distances: list[float] = []
    encoder_deltas: list[int] = []

    for filename, measured_distance_m in files_and_data:
        path = data_dir / filename
        if not path.exists():
            print(f"Skipping missing file: {path}")
            continue

        try:
            data_dict = load_log_dict(path)
        except Exception as e:  # noqa: BLE001
            print(f"Skipping unreadable file: {path} ({e})")
            continue

        delta_ticks = encoder_delta_ticks(data_dict)
        measured_distances.append(float(measured_distance_m))
        encoder_deltas.append(int(delta_ticks))

    if len(encoder_deltas) < 2:
        raise SystemExit("Not enough valid trials to fit a trendline.")

    ticks_list = [float(x) for x in encoder_deltas]
    slope_m_per_tick = fit_slope_m_per_tick(ticks_list, measured_distances)

    predicted_distances = [slope_m_per_tick * ticks for ticks in ticks_list]
    squared_errors = [
        (measured_distances[i] - predicted_distances[i]) ** 2 for i in range(len(measured_distances))
    ]
    rmse = (sum(squared_errors) / len(squared_errors)) ** 0.5

    # Constant variance model: use average squared residual as sigma_s^2.
    # Exclude detected variance outliers from the mean calculation.
    outlier_indices = detect_outlier_indices(squared_errors, max_outliers=1)
    outlier_index_set = set(outlier_indices)
    if outlier_indices:
        inlier_squared_errors = [e for i, e in enumerate(squared_errors) if i not in outlier_index_set]
    else:
        inlier_squared_errors = squared_errors
    variance_mean = sum(inlier_squared_errors) / len(inlier_squared_errors)

    print("Encoder → distance trendline:")
    print(f"  fit: distance_m = {slope_m_per_tick:.6g} * ticks")
    print(f"  slope (meters per tick): {slope_m_per_tick:.6g}")
    if slope_m_per_tick != 0:
        print(f"  ticks per meter: {1.0 / slope_m_per_tick:.6g}")
    print("Encoder ticks → variance trendline:")
    print(f"  fit: sigma_s^2 = {variance_mean:.6g} (average)")
    for outlier_index in outlier_indices:
        print(
            f"  excluded outlier at ticks={ticks_list[outlier_index]:.0f}, "
            f"variance={squared_errors[outlier_index]:.6g}"
        )

    # Plots (optional; environments used for grading typically have matplotlib installed)
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ModuleNotFoundError:
        print("Note: matplotlib not installed; skipping plots.")
        return

    # Single figure with stacked subplots:
    # Top: measured distance vs encoder count
    # Bottom: variance vs encoder count
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=False)

    # Top subplot: measured distance vs encoder count, with fitted trendline
    if not outlier_indices:
        top_data_handle, = ax1.plot(ticks_list, measured_distances, "ko", label="Measured data")
    else:
        inlier_ticks = [t for i, t in enumerate(ticks_list) if i not in outlier_index_set]
        inlier_distances = [d for i, d in enumerate(measured_distances) if i not in outlier_index_set]
        top_data_handle, = ax1.plot(inlier_ticks, inlier_distances, "ko", label="Measured data")
        ax1.plot(
            [ticks_list[i] for i in outlier_indices],
            [measured_distances[i] for i in outlier_indices],
            "ro",
            markersize=7,
        )
    max_ticks = max(ticks_list)
    line_x = [0.0, max_ticks]
    line_y = [slope_m_per_tick * x for x in line_x]
    equation_label = rf"y = ({slope_m_per_tick:.4g})x, RMSE = {rmse:.4g} m"
    top_fit_handle, = ax1.plot(line_x, line_y, "b-", label=equation_label)
    ax1.set_title("Measured Distance vs. Encoder Measurement")
    ax1.set_xlabel("Encoder count delta (ticks)")
    ax1.set_ylabel("Measured Distance (m)")
    ax1.legend([top_data_handle, top_fit_handle], ["Measured data", equation_label])
    ax1.axhline(0.0, color="0.4", linewidth=1.0)
    ax1.axvline(0.0, color="0.4", linewidth=1.0)
    ax1.set_xlim(left=0.0)
    ax1.set_ylim(bottom=0.0)
    ax1.set_box_aspect(0.5)
    ax1.grid(True)

    # Bottom subplot: variance (squared error) vs encoder ticks, with average variance
    if not outlier_indices:
        ax2.plot(ticks_list, squared_errors, "ko")
    else:
        inlier_ticks = [t for i, t in enumerate(ticks_list) if i not in outlier_index_set]
        inlier_variance = [v for i, v in enumerate(squared_errors) if i not in outlier_index_set]
        ax2.plot(inlier_ticks, inlier_variance, "ko")
        ax2.plot(
            [ticks_list[i] for i in outlier_indices],
            [squared_errors[i] for i in outlier_indices],
            "ro",
            markersize=7,
        )
    ax2.axhline(variance_mean, color="r", linewidth=1.5)
    ax2.set_title("Estimated Distance Variance vs Encoder Ticks")
    ax2.set_xlabel("Encoder count delta (ticks)")
    ax2.set_ylabel(r"Estimated Variance $\sigma_s^2$ (m$^2$)")
    mean_label = rf"Average variance = {variance_mean:.4g} m$^2$"
    if not outlier_indices:
        ax2.legend([r"$(s - \hat{s})^2$", mean_label])
    else:
        ax2.legend(["Inliers", "Outliers", mean_label])
    ax2.axhline(0.0, color="0.4", linewidth=1.0)
    ax2.axvline(0.0, color="0.4", linewidth=1.0)
    ax2.set_xlim(left=0.0)
    ax2.set_ylim(0.0, 0.002)
    ax2.set_box_aspect(0.5)
    ax2.grid(True)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
