from __future__ import annotations

import pickle
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

def fit_quadratic(x: list[float], y: list[float]) -> tuple[float, float, float]:
    """
    Fit: y ≈ a*x^2 + b*x + c (least squares).
    Returns (a, b, c).
    """

    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    if len(x) < 3:
        raise ValueError("need at least 3 points to fit a quadratic")

    sx0 = float(len(x))
    sx1 = sum(xi for xi in x)
    sx2 = sum(xi * xi for xi in x)
    sx3 = sum(xi * xi * xi for xi in x)
    sx4 = sum(xi * xi * xi * xi for xi in x)

    sy0 = sum(yi for yi in y)
    sy1 = sum(x[i] * y[i] for i in range(len(x)))
    sy2 = sum((x[i] * x[i]) * y[i] for i in range(len(x)))

    A = [
        [sx4, sx3, sx2],
        [sx3, sx2, sx1],
        [sx2, sx1, sx0],
    ]
    b = [sy2, sy1, sy0]

    for col in range(3):
        pivot_row = max(range(col, 3), key=lambda r: abs(A[r][col]))
        if A[pivot_row][col] == 0:
            raise ValueError("cannot fit quadratic: singular matrix")
        if pivot_row != col:
            A[col], A[pivot_row] = A[pivot_row], A[col]
            b[col], b[pivot_row] = b[pivot_row], b[col]

        pivot = A[col][col]
        for j in range(col, 3):
            A[col][j] /= pivot
        b[col] /= pivot

        for r in range(3):
            if r == col:
                continue
            factor = A[r][col]
            for j in range(col, 3):
                A[r][j] -= factor * A[col][j]
            b[r] -= factor * b[col]

    a, b_coef, c = b[0], b[1], b[2]
    return float(a), float(b_coef), float(c)


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

    # Fit a non-constant variance model: sigma_s^2 = f_ss(s)
    # Per lab instructions, use the squared difference as an estimate of sigma_s^2.
    a_var, b_var, c_var = fit_quadratic(measured_distances, squared_errors)

    print("Encoder → distance trendline:")
    print(f"  fit: distance_m = {slope_m_per_tick:.6g} * ticks")
    print(f"  slope (meters per tick): {slope_m_per_tick:.6g}")
    if slope_m_per_tick != 0:
        print(f"  ticks per meter: {1.0 / slope_m_per_tick:.6g}")
    print("Distance → variance trendline:")
    print(f"  fit: sigma_s^2 = {a_var:.6g} * s^2 + {b_var:.6g} * s + {c_var:.6g}")

    # Plots (optional; environments used for grading typically have matplotlib installed)
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ModuleNotFoundError:
        print("Note: matplotlib not installed; skipping plots.")
        return

    # Figure 1: measured distance vs encoder count, with fitted trendline
    plt.figure()
    plt.plot(ticks_list, measured_distances, "ko")
    max_ticks = max(ticks_list)
    line_x = [0.0, max_ticks]
    line_y = [slope_m_per_tick * x for x in line_x]
    plt.plot(line_x, line_y, "b-")
    plt.title("Measured Distance vs. Encoder Measurement")
    plt.xlabel("Encoder count delta (ticks)")
    plt.ylabel("Measured Distance (m)")
    equation_label = rf"y = ({slope_m_per_tick:.4g})x, RMSE = {rmse:.4g} m"
    plt.legend(["Measured data", equation_label])
    plt.axhline(0.0, color="0.4", linewidth=1.0)
    plt.axvline(0.0, color="0.4", linewidth=1.0)
    plt.xlim(left=0.0)
    plt.ylim(bottom=0.0)
    plt.gca().set_box_aspect(0.5)
    plt.grid(True)

    # Figure 2: variance (squared error) vs measured distance, with quadratic fit
    plt.figure()
    plt.plot(measured_distances, squared_errors, "ko")
    max_s = max(max(measured_distances), max(predicted_distances), 0.0)
    s_grid = [i * max_s / 200.0 for i in range(201)]
    var_fit = [a_var * s * s + b_var * s + c_var for s in s_grid]
    plt.plot(s_grid, var_fit, "r-")
    plt.title("Estimated Distance Variance vs Distance")
    plt.xlabel("Measured Distance (m)")
    plt.ylabel(r"Estimated Variance $\sigma_s^2$ (m$^2$)")
    plt.legend([r"$(s - \hat{s})^2$", r"Quadratic fit $f_{ss}(s)$"])
    plt.axhline(0.0, color="0.4", linewidth=1.0)
    plt.axvline(0.0, color="0.4", linewidth=1.0)
    plt.xlim(left=0.0)
    plt.ylim(bottom=0.0)
    plt.gca().set_box_aspect(0.5)
    plt.grid(True)

    plt.show()


if __name__ == "__main__":
    main()
