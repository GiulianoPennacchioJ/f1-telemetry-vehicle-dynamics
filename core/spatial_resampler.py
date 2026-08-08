import logging
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class SpatialResampler:
    """
    Transforms time-series telemetry data into a fixed-step spatial domain (s = 1.0 meter grid).
    This eliminates temporal asynchronous sampling artifacts between different cars/laps.
    """

    def __init__(self, step_size_meters: float = 1.0):
        """
        Parameters:
        -----------
        step_size_meters : float
            Spatial grid resolution in meters (default: 1.0 m).
        """
        self.step_size = step_size_meters

    def _compute_cumulative_distance(self, telemetry: pd.DataFrame) -> np.ndarray:
        """
        Integrates longitudinal speed over time to generate the spatial coordinate s(t).
        Uses trapezoidal integration: s(t) = ∫ v(τ) dτ.
        """
        # Convert speed from km/h to m/s
        speed_ms = telemetry['Speed'].values / 3.6
        
        # Calculate time delta in seconds using SessionTime
        time_sec = telemetry['SessionTime'].dt.total_seconds().values
        dt = np.diff(time_sec, prepend=time_sec[0])
        dt[0] = 0.0  # Initial step has zero duration

        # Cumulative distance array in meters
        distance_m = np.cumsum(speed_ms * dt)
        return distance_m

    def resample_lap(self, telemetry: pd.DataFrame) -> pd.DataFrame:
        """
        Resamples all telemetry channels onto a uniform spatial grid.

        Parameters:
        -----------
        telemetry : pd.DataFrame
            Raw telemetry DataFrame containing time-based samples.

        Returns:
        --------
        resampled_df : pd.DataFrame
            Spatially uniform telemetry DataFrame indexed by distance 's' (meters).
        """
        df = telemetry.copy()
        
        # Compute distance vector
        df['Distance'] = self._compute_cumulative_distance(df)

        # Handle duplicate distance entries (if car is stationary or data artifact occurs)
        df = df.drop_duplicates(subset=['Distance'], keep='first')

        s_raw = df['Distance'].values
        s_max = np.floor(s_raw[-1])
        
        # Create uniform spatial grid: s = [0, 1, 2, ..., s_max]
        s_grid = np.arange(0, s_max + self.step_size, self.step_size)

        resampled_data = {'Distance': s_grid}

        # Continuous channels interpolated linearly
        continuous_cols = ['Speed', 'RPM', 'Throttle', 'Brake', 'X', 'Y', 'Z']
        for col in continuous_cols:
            if col in df.columns:
                f_linear = interp1d(s_raw, df[col].values, kind='linear', fill_value="extrapolate")
                resampled_data[col] = f_linear(s_grid)

        # Discrete/State channels interpolated using nearest/previous neighbor
        discrete_cols = ['Gear', 'DRS']
        for col in discrete_cols:
            if col in df.columns:
                f_nearest = interp1d(s_raw, df[col].values, kind='previous', fill_value="extrapolate")
                resampled_data[col] = f_nearest(s_grid)

        resampled_df = pd.DataFrame(resampled_data)
        logging.info(f"Spatial resampling complete. Input: {len(df)} samples -> Output: {len(resampled_df)} spatial points (Δs = {self.step_size}m).")
        
        return resampled_df


if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Add project root directory to sys.path to resolve internal package imports
    sys.path.append(str(Path(__file__).resolve().parent.parent))

    from core.data_loader import F1DataLoader

    loader = F1DataLoader()
    session = loader.load_session(2025, 'Monza', 'Q')
    nor_lap = loader.get_driver_fastest_lap(session, 'NOR')
    raw_telemetry = loader.extract_raw_telemetry(nor_lap)

    resampler = SpatialResampler(step_size_meters=1.0)
    spatial_telemetry = resampler.resample_lap(raw_telemetry)
    
    print("\n==========================================")
    print(" SPATIAL RESAMPLING SUCCESSFUL")
    print("==========================================")
    print(spatial_telemetry.head())