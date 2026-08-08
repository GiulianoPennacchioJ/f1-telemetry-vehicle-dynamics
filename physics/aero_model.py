import os
import json
import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class AeroModel:
    """
    Computes downforce (F_z), drag forces (F_x), and effective aerodynamic coefficients
    accounting for DRS activation state according to 2025 F1 technical specifications.
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
        """Loads physical and aerodynamic constants from JSON configuration."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found at: {self.config_path}")

        with open(self.config_path, "r") as f:
            config = json.load(f)

        # Vehicle & Environmental constants
        self.rho = config["vehicle"]["air_density_kg_m3"]
        self.area = config["vehicle"]["frontal_area_m2"]
        
        # Aerodynamic coefficients
        self.cl = config["aerodynamics"]["cl_base"]
        self.cd_closed = config["aerodynamics"]["cd_base_drs_closed"]
        self.cd_open = config["aerodynamics"]["cd_drs_open"]

        logging.info("AeroModel parameters loaded successfully from config.")

    def is_drs_active(self, drs_signal: float) -> bool:
        """
        Determines if the DRS is open based on the telemetry channel signal.
        In FastF1 telemetry, values >= 10 (or odd integers > 8) indicate open DRS state.
        """
        return drs_signal in [10, 12, 14] or drs_signal >= 10.0

    def compute_downforce(self, speed_ms: np.ndarray) -> np.ndarray:
        """
        Calculates vertical downforce: F_z = 0.5 * ρ * C_L * A * v^2 [N]
        """
        return 0.5 * self.rho * self.cl * self.area * (speed_ms ** 2)

    def compute_drag_force(self, speed_ms: np.ndarray, drs_signals: np.ndarray) -> np.ndarray:
        """
        Calculates longitudinal aerodynamic drag force considering DRS state:
        F_drag = 0.5 * ρ * C_D(DRS) * A * v^2 [N]
        """
        # Vectorized assignment of Cd depending on DRS state
        cd_array = np.where(
            np.isin(drs_signals, [10, 12, 14]) | (drs_signals >= 10.0),
            self.cd_open,
            self.cd_closed
        )
        return 0.5 * self.rho * cd_array * self.area * (speed_ms ** 2)

    def process_telemetry_aero(self, spatial_df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the aerodynamic model to a resampled telemetry DataFrame.

        Parameters:
        -----------
        spatial_df : pd.DataFrame
            Telemetry resampled on uniform spatial grid (Δs = 1.0 m) containing 'Speed' and 'DRS'.

        Returns:
        --------
        aero_df : pd.DataFrame
            DataFrame augmented with 'Fz_Downforce_N', 'Fx_Drag_N', 'Cd_Apparent', and 'Aero_Efficiency_L_D'.
        """
        df = spatial_df.copy()

        # Speed conversion to m/s (use smoothed speed if available)
        speed_col = 'Speed_Smoothed_kmh' if 'Speed_Smoothed_kmh' in df.columns else 'Speed'
        v_ms = df[speed_col].values / 3.6
        drs_signals = df['DRS'].values

        # Compute Aerodynamic Forces
        fz_downforce = self.compute_downforce(v_ms)
        fx_drag = self.compute_drag_force(v_ms, drs_signals)

        # Compute dynamic Cd and L/D ratio
        cd_array = np.where(
            np.isin(drs_signals, [10, 12, 14]) | (drs_signals >= 10.0),
            self.cd_open,
            self.cd_closed
        )
        efficiency_ld = self.cl / cd_array

        # Assign output channels
        df['Fz_Downforce_N'] = fz_downforce
        df['Fx_Drag_N'] = fx_drag
        df['Cd_Apparent'] = cd_array
        df['Aero_Efficiency_L_D'] = efficiency_ld

        logging.info("Aerodynamic forces (Fz, Fx) computed successfully across spatial grid.")
        return df


if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Add project root directory to sys.path to resolve internal package imports
    sys.path.append(str(Path(__file__).resolve().parent.parent))

    from core.data_loader import F1DataLoader
    from core.spatial_resampler import SpatialResampler
    from core.signal_processing import SignalProcessor

    # Load and process lap data for validation
    loader = F1DataLoader()
    session = loader.load_session(2025, 'Monza', 'Q')
    nor_lap = loader.get_driver_fastest_lap(session, 'NOR')
    
    raw_df = loader.extract_raw_telemetry(nor_lap)
    resampler = SpatialResampler(step_size_meters=1.0)
    spatial_df = resampler.resample_lap(raw_df)

    processor = SignalProcessor(window_length=15, poly_order=3)
    kinematic_df = processor.compute_kinematics(spatial_df)

    # Instantiate and execute AeroModel
    aero = AeroModel(config_path="config/car_parameters.json")
    aero_df = aero.process_telemetry_aero(kinematic_df)

    print("\n==========================================")
    print(" AERODYNAMIC MODEL SUCCESSFUL")
    print("==========================================")
    print(aero_df[['Distance', 'Speed', 'DRS', 'Cd_Apparent', 'Fz_Downforce_N', 'Fx_Drag_N', 'Aero_Efficiency_L_D']].head(10))