import os
import json
import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class TireModel:
    """
    Models vertical load distribution, dynamic longitudinal load transfer,
    and friction coefficient utilization (μ_util) across the spatial telemetry grid.
    """

    def __init__(self, config_path: str = "config/car_parameters.json"):
        """
        Parameters:
        -----------
        config_path : str
            Path to the JSON file containing vehicle physical parameters.
        """
        self.config_path = config_path
        self._load_parameters()

    def _load_parameters(self) -> None:
        """Loads vehicle weight and balance constants from JSON configuration."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found at: {self.config_path}")

        with open(self.config_path, "r") as f:
            config = json.load(f)

        self.mass = config["vehicle"]["mass_kg"]
        self.g = config["vehicle"]["gravity_m_s2"]
        self.weight_static_N = self.mass * self.g
        self.mu_peak = config["tires"]["peak_friction_coefficient_mu"]
        self.rear_bias = config["tires"]["mass_distribution_rear_pct"] / 100.0

        logging.info("TireModel parameters loaded successfully from config.")

    def compute_total_vertical_load(self, fz_downforce: np.ndarray) -> np.ndarray:
        """
        Calculates total normal vertical force: F_z,total = Mass * g + F_z,downforce [N]
        """
        return self.weight_static_N + fz_downforce

    def compute_in_plane_force(self, ax_ms2: np.ndarray, ay_ms2: np.ndarray) -> np.ndarray:
        """
        Calculates total magnitude of planar inertial force: F_xy = Mass * sqrt(a_x^2 + a_y^2) [N]
        """
        return self.mass * np.sqrt(ax_ms2 ** 2 + ay_ms2 ** 2)

    def compute_friction_utilization(self, f_xy_N: np.ndarray, fz_total_N: np.ndarray) -> np.ndarray:
        """
        Calculates required friction utilization factor: μ_util = F_xy / F_z,total
        """
        # Guard against division by zero
        safe_fz = np.maximum(fz_total_N, 1.0)
        return f_xy_N / safe_fz

    def process_telemetry_tires(self, aero_df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies tire dynamics and load calculations to the aerodynamic-augmented DataFrame.

        Parameters:
        -----------
        aero_df : pd.DataFrame
            DataFrame from AeroModel containing 'Fz_Downforce_N', acceleration channels, etc.

        Returns:
        --------
        tire_df : pd.DataFrame
            DataFrame augmented with 'Fz_Total_N', 'F_xy_InPlane_N', and 'Mu_Utilized'.
        """
        df = aero_df.copy()

        # Robust extraction of acceleration channels (m/s^2)
        if 'a_x_ms2' in df.columns:
            ax = df['a_x_ms2'].values
        elif 'a_x' in df.columns:
            ax = df['a_x'].values
        elif 'a_x_g' in df.columns:
            ax = df['a_x_g'].values * self.g
        else:
            ax = np.zeros(len(df))

        if 'a_y_ms2' in df.columns:
            ay = df['a_y_ms2'].values
        elif 'a_y' in df.columns:
            ay = df['a_y'].values
        elif 'a_y_g' in df.columns:
            ay = df['a_y_g'].values * self.g
        else:
            ay = np.zeros(len(df))

        fz_downforce = df['Fz_Downforce_N'].values

        # Compute vertical loads and planar forces
        fz_total = self.compute_total_vertical_load(fz_downforce)
        f_xy = self.compute_in_plane_force(ax, ay)
        mu_util = self.compute_friction_utilization(f_xy, fz_total)

        # Assign output channels
        df['Fz_Total_N'] = fz_total
        df['F_xy_InPlane_N'] = f_xy
        df['Mu_Utilized'] = mu_util

        logging.info("Tire load distribution and friction utilization (μ_util) computed successfully.")
        return df


if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Add project root directory to sys.path to resolve internal package imports
    sys.path.append(str(Path(__file__).resolve().parent.parent))

    from core.data_loader import F1DataLoader
    from core.spatial_resampler import SpatialResampler
    from core.signal_processing import SignalProcessor
    from physics.aero_model import AeroModel

    # Execution pipeline up to AeroModel
    loader = F1DataLoader()
    session = loader.load_session(2025, 'Monza', 'Q')
    nor_lap = loader.get_driver_fastest_lap(session, 'NOR')
    
    raw_df = loader.extract_raw_telemetry(nor_lap)
    resampler = SpatialResampler(step_size_meters=1.0)
    spatial_df = resampler.resample_lap(raw_df)

    processor = SignalProcessor(window_length=15, poly_order=3)
    kinematic_df = processor.compute_kinematics(spatial_df)

    aero = AeroModel(config_path="config/car_parameters.json")
    aero_df = aero.process_telemetry_aero(kinematic_df)

    # Instantiate and execute TireModel
    tire_model = TireModel(config_path="config/car_parameters.json")
    tire_df = tire_model.process_telemetry_tires(aero_df)

    print("\n==========================================")
    print(" TIRE MODEL SUCCESSFUL")
    print("==========================================")
    
    # Dynamic selection of columns present in DataFrame for printing
    cols_to_show = ['Distance', 'Speed', 'Fz_Total_N', 'F_xy_InPlane_N', 'Mu_Utilized']
    for ax_col in ['a_x_ms2', 'a_x_g', 'a_x']:
        if ax_col in tire_df.columns:
            cols_to_show.insert(2, ax_col)
            break
    for ay_col in ['a_y_ms2', 'a_y_g', 'a_y']:
        if ay_col in tire_df.columns:
            cols_to_show.insert(3, ay_col)
            break

    print(tire_df[cols_to_show].head(10))