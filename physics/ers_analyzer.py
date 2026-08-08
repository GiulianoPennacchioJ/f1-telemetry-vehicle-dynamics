import os
import json
import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class ERSAnalyzer:
    """
    Analyzes ERS (Energy Recovery System) deployment, harvesting, and clipping
    in compliance with 2025 F1 Technical Regulations (MGU-K max 120 kW).
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
        """Loads powertrain and vehicle parameters from JSON configuration."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found at: {self.config_path}")

        with open(self.config_path, "r") as f:
            config = json.load(f)

        self.mass = config["vehicle"]["mass_kg"]
        self.max_ice_power_kw = config["powertrain_2025"]["max_ice_power_kw"]
        self.max_mguk_power_kw = config["powertrain_2025"]["max_mguk_power_kw"]
        self.mguk_harvest_limit_mj = config["powertrain_2025"]["mguk_harvest_limit_mj_lap"]

        logging.info("ERSAnalyzer parameters loaded successfully from config.")

    def compute_wheel_power(self, speed_ms: np.ndarray, ax_ms2: np.ndarray, fx_drag: np.ndarray) -> np.ndarray:
        """
        Calculates net required mechanical propulsion power at wheels:
        P_wheels = (m * a_x + F_drag) * v [kW]
        """
        # Inertial force + Aerodynamic drag resistance
        f_propulsion = (self.mass * ax_ms2) + fx_drag
        power_watts = f_propulsion * speed_ms
        return power_watts / 1000.0  # Convert Watts to kW

    def estimate_mguk_power(self, power_wheels_kw: np.ndarray, throttle: np.ndarray) -> np.ndarray:
        """
        Estimates MGU-K power contribution bounded by 2025 regulation limit (120 kW).
        """
        # Base ICE capacity baseline
        ice_baseline_kw = self.max_ice_power_kw * (throttle / 100.0)
        
        # Power delta required beyond internal combustion engine
        delta_power_kw = power_wheels_kw - ice_baseline_kw
        
        # MGU-K deployment bounded between 0 kW and +120 kW
        mguk_deployment = np.clip(delta_power_kw, 0.0, self.max_mguk_power_kw)
        
        # Zero out deployment when throttle is not pressed
        return np.where(throttle > 50.0, mguk_deployment, 0.0)

    def detect_clipping_events(self, throttle: np.ndarray, speed_kmh: np.ndarray, ax_g: np.ndarray) -> np.ndarray:
        """
        Detects ERS clipping (battery exhaustion on high-speed straights).
        Criteria: Throttle == 100%, Speed > 280 km/h, and a_x drops near/below 0 G.
        """
        is_full_throttle = throttle >= 98.0
        is_high_speed = speed_kmh >= 280.0
        is_accel_drop = ax_g <= 0.05

        return is_full_throttle & is_high_speed & is_accel_drop

    def process_telemetry_ers(self, gg_df: pd.DataFrame) -> pd.DataFrame:
        """
        Processes resampled telemetry to extract ERS power, deployment state, and clipping.

        Parameters:
        -----------
        gg_df : pd.DataFrame
            Telemetry DataFrame containing kinematic accelerations and aerodynamic forces.

        Returns:
        --------
        ers_df : pd.DataFrame
            DataFrame augmented with 'Power_Wheels_kW', 'MGU_K_Power_kW', and 'Is_Clipping'.
        """
        df = gg_df.copy()

        # Extract required channels
        v_ms = (df['Speed_Smoothed_kmh'].values if 'Speed_Smoothed_kmh' in df.columns else df['Speed'].values) / 3.6
        v_kmh = v_ms * 3.6
        
        ax_ms2 = df['a_x'].values if 'a_x' in df.columns else df.get('a_x_ms2', np.zeros(len(df))).values
        ax_g = df['a_x_g'].values if 'a_x_g' in df.columns else ax_ms2 / 9.81
        
        fx_drag = df['Fx_Drag_N'].values if 'Fx_Drag_N' in df.columns else np.zeros(len(df))
        throttle = df['Throttle'].values if 'Throttle' in df.columns else np.full(len(df), 100.0)

        # Compute Power channels
        p_wheels_kw = self.compute_wheel_power(v_ms, ax_ms2, fx_drag)
        p_mguk_kw = self.estimate_mguk_power(p_wheels_kw, throttle)
        is_clipping = self.detect_clipping_events(throttle, v_kmh, ax_g)

        # Assign output channels
        df['Power_Wheels_kW'] = p_wheels_kw
        df['MGU_K_Power_kW'] = p_mguk_kw
        df['Is_Clipping'] = is_clipping

        logging.info("2025 ERS power dynamics and clipping detection processed successfully.")
        return df


if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Add project root directory to sys.path
    sys.path.append(str(Path(__file__).resolve().parent.parent))

    from core.data_loader import F1DataLoader
    from core.spatial_resampler import SpatialResampler
    from core.signal_processing import SignalProcessor
    from physics.aero_model import AeroModel
    from physics.tire_model import TireModel
    from physics.gg_diagram import GGDiagramAnalyzer

    # Execution pipeline up to GGDiagramAnalyzer
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

    tire = TireModel(config_path="config/car_parameters.json")
    tire_df = tire.process_telemetry_tires(aero_df)

    gg_analyzer = GGDiagramAnalyzer()
    gg_df, _ = gg_analyzer.process_telemetry_gg(tire_df)

    # Instantiate and execute ERSAnalyzer
    ers_analyzer = ERSAnalyzer(config_path="config/car_parameters.json")
    ers_df = ers_analyzer.process_telemetry_ers(gg_df)

    print("\n==========================================")
    print(" 2025 ERS ANALYSIS SUCCESSFUL")
    print("==========================================")
    print("Max Wheel Power Required: ", f"{ers_df['Power_Wheels_kW'].max():.1f} kW")
    print("Max MGU-K Deployment:     ", f"{ers_df['MGU_K_Power_kW'].max():.1f} kW (Limit: 120.0 kW)")
    print("Total Clipping Distance:  ", f"{ers_df['Is_Clipping'].sum():.1f} meters")
    print("\nSample Output at Top Speed:")
    print(ers_df[['Distance', 'Speed', 'Throttle', 'Power_Wheels_kW', 'MGU_K_Power_kW', 'Is_Clipping']].head(10))