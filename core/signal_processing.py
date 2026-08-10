import logging
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class SignalProcessor:
    """
    Applies Savitzky-Golay filtering on spatially uniform telemetry 
    and computes longitudinal (a_x) and lateral (a_y) accelerations.
    """

    def __init__(self, window_length: int = 15, poly_order: int = 3):
        """
        Parameters:
        -----------
        window_length : int
            Window length for Savitzky-Golay filter (must be an odd integer).
        poly_order : int
            Polynomial order for fitting local data points.
        """
        if window_length % 2 == 0:
            window_length += 1  # Force window length to be odd
            
        self.window_length = window_length
        self.poly_order = poly_order
        self.g_acc = 9.81  # Gravitational acceleration constant [m/s^2]

    def filter_signal(self, signal: np.ndarray) -> np.ndarray:
        """Applies 1D Savitzky-Golay smoothing to a spatial signal."""
        return savgol_filter(signal, window_length=self.window_length, polyorder=self.poly_order)

    def compute_kinematics(self, spatial_df: pd.DataFrame, step_size: float = 1.0) -> pd.DataFrame:
        """
        Computes analytical derivatives and acceleration g-forces (a_x, a_y).

        Parameters:
        -----------
        spatial_df : pd.DataFrame
            DataFrame resampled on a uniform spatial grid Δs.
        step_size : float
            Spatial grid spacing in meters.

        Returns:
        --------
        processed_df : pd.DataFrame
            DataFrame augmented with smoothed signals and acceleration channels in G-units.
        """
        df = spatial_df.copy()

        # 1. Smooth Longitudinal Velocity [m/s]
        v_ms = df['Speed'].values / 3.6
        v_smooth = self.filter_signal(v_ms)
        df['Speed_Smoothed_kmh'] = v_smooth * 3.6

        # 2. Longitudinal Acceleration: a_x = v * (dv/ds)
        dv_ds = savgol_filter(
            v_ms, 
            window_length=self.window_length, 
            polyorder=self.poly_order, 
            deriv=1, 
            delta=step_size
        )
        a_x_ms2 = v_smooth * dv_ds
        df['a_x'] = a_x_ms2 / self.g_acc  # Convert to G-force

        # 3. Lateral Acceleration via Path Curvature: κ = |X'Y'' - Y'X''| / (X'^2 + Y'^2)^(3/2)
        if 'X' in df.columns and 'Y' in df.columns:
            x = df['X'].values
            y = df['Y'].values

            # Derivatives of spatial coordinates w.r.t distance 's'
            dx_ds = savgol_filter(x, self.window_length, self.poly_order, deriv=1, delta=step_size)
            ddx_ds2 = savgol_filter(x, self.window_length, self.poly_order, deriv=2, delta=step_size)
            
            dy_ds = savgol_filter(y, self.window_length, self.poly_order, deriv=1, delta=step_size)
            ddy_ds2 = savgol_filter(y, self.window_length, self.poly_order, deriv=2, delta=step_size)

            # Menger curvature formulation (Rimosso np.abs per preservare il segno della curva!)
            numerator = (dx_ds * ddy_ds2 - dy_ds * ddx_ds2)
            denominator = np.power(dx_ds**2 + dy_ds**2, 1.5)
            
            # Avoid division by zero on straightaways
            kappa = np.where(denominator > 1e-6, numerator / denominator, 0.0)

            # Lateral Acceleration: a_y = v^2 * κ
            a_y_ms2 = (v_smooth ** 2) * kappa
            df['a_y'] = a_y_ms2 / self.g_acc  # Convert to G-force
            df['Curvature'] = kappa
        else:
            logging.warning("GPS X, Y channels not found. Skipping lateral acceleration (a_y) calculation.")
            df['a_y'] = 0.0

        logging.info("Kinematic processing complete: a_x and a_y G-forces computed successfully.")
        return df


if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Add project root directory to sys.path to resolve internal package imports
    sys.path.append(str(Path(__file__).resolve().parent.parent))

    from core.data_loader import F1DataLoader
    from core.spatial_resampler import SpatialResampler

    loader = F1DataLoader()
    session = loader.load_session(2025, 'Monza', 'Q')
    nor_lap = loader.get_driver_fastest_lap(session, 'NOR')
    
    raw_df = loader.extract_raw_telemetry(nor_lap)
    resampler = SpatialResampler(step_size_meters=1.0)
    spatial_df = resampler.resample_lap(raw_df)

    processor = SignalProcessor(window_length=15, poly_order=3)
    final_df = processor.compute_kinematics(spatial_df)

    print("\n==========================================")
    print(" SIGNAL PROCESSING & KINEMATICS SUCCESSFUL")
    print("==========================================")
    print(final_df[['Distance', 'Speed', 'Speed_Smoothed_kmh', 'a_x', 'a_y']].head(10))