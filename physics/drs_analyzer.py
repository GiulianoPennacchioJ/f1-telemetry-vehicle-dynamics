import os
import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class DRSAnalyzer:
    """
    Analyzes DRS (Drag Reduction System) activation zones, quantifies drag force reduction (ΔFx),
    and computes equivalent power savings according to 2025 F1 regulations.
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
        """Loads aerodynamic coefficients from JSON configuration."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found at: {self.config_path}")

        with open(self.config_path, "r") as f:
            config = json.load(f)

        self.rho = config["vehicle"]["air_density_kg_m3"]
        self.area = config["vehicle"]["frontal_area_m2"]
        self.cd_closed = config["aerodynamics"]["cd_base_drs_closed"]
        self.cd_open = config["aerodynamics"]["cd_drs_open"]
        self.delta_cd = self.cd_closed - self.cd_open

        logging.info("DRSAnalyzer parameters loaded successfully from config.")

    def compute_drag_delta(self, speed_ms: np.ndarray, drs_signals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculates instantaneous drag reduction force ΔFx [N] and equivalent power gain ΔP [kW]
        when DRS is active.
        """
        is_open = np.isin(drs_signals, [10, 12, 14]) | (drs_signals >= 10.0)
        
        # Aerodynamic force delta: ΔF_drag = 0.5 * ρ * ΔC_D * A * v^2
        delta_fx_N = np.where(is_open, 0.5 * self.rho * self.delta_cd * self.area * (speed_ms ** 2), 0.0)
        
        # Equivalent power saved: ΔP = (ΔFx * v) / 1000 [kW]
        delta_power_kw = (delta_fx_N * speed_ms) / 1000.0
        
        return delta_fx_N, delta_power_kw

    def detect_drs_activation_zones(self, spatial_df: pd.DataFrame) -> pd.DataFrame:
        """
        Identifies spatial start, end, and length of contiguous DRS activation zones.
        """
        drs_signals = spatial_df['DRS'].values
        distances = spatial_df['Distance'].values
        is_open = (np.isin(drs_signals, [10, 12, 14]) | (drs_signals >= 10.0)).astype(int)

        # Find state transitions (0 -> 1 start, 1 -> 0 end)
        diff = np.diff(np.pad(is_open, (1, 1), 'constant'))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0] - 1

        zones = []
        for i, (start_idx, end_idx) in enumerate(zip(starts, ends)):
            start_dist = distances[start_idx]
            end_dist = distances[end_idx]
            length_m = end_dist - start_dist
            zones.append({
                'Zone_ID': i + 1,
                'Start_Distance_m': start_dist,
                'End_Distance_m': end_dist,
                'Length_m': length_m
            })

        return pd.DataFrame(zones)

    def process_telemetry_drs(self, ers_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Processes resampled telemetry to compute DRS delta forces, power gain, and zone mapping.

        Parameters:
        -----------
        ers_df : pd.DataFrame
            Telemetry DataFrame from ERSAnalyzer.

        Returns:
        --------
        drs_df : pd.DataFrame
            DataFrame augmented with 'DRS_Active', 'Delta_Fx_Drag_N', and 'DRS_Power_Gain_kW'.
        zones_df : pd.DataFrame
            Summary DataFrame of active DRS zones along the track.
        """
        df = ers_df.copy()

        # Extract required signals
        v_ms = (df['Speed_Smoothed_kmh'].values if 'Speed_Smoothed_kmh' in df.columns else df['Speed'].values) / 3.6
        drs_signals = df['DRS'].values

        is_open = np.isin(drs_signals, [10, 12, 14]) | (drs_signals >= 10.0)
        delta_fx_N, delta_power_kw = self.compute_drag_delta(v_ms, drs_signals)
        zones_df = self.detect_drs_activation_zones(df)

        # Assign output channels
        df['DRS_Active'] = is_open
        df['Delta_Fx_Drag_N'] = delta_fx_N
        df['DRS_Power_Gain_kW'] = delta_power_kw

        logging.info("DRS aerodynamic delta and track activation zones processed successfully.")
        return df, zones_df


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
    from physics.ers_analyzer import ERSAnalyzer

    # Complete physics pipeline execution
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

    ers_analyzer = ERSAnalyzer(config_path="config/car_parameters.json")
    ers_df = ers_analyzer.process_telemetry_ers(gg_df)

    # Instantiate and execute DRSAnalyzer
    drs_analyzer = DRSAnalyzer(config_path="config/car_parameters.json")
    final_physics_df, drs_zones = drs_analyzer.process_telemetry_drs(ers_df)

    print("\n==========================================")
    print(" DRS ANALYSIS SUCCESSFUL")
    print("==========================================")
    print("\nDRS Activation Zones Summary:")
    print(drs_zones.to_string(index=False))

    print("\nMax Aerodynamic Drag Reduction: ", f"{final_physics_df['Delta_Fx_Drag_N'].max():.1f} N")
    print("Max Equivalent Power Saved:     ", f"{final_physics_df['DRS_Power_Gain_kW'].max():.1f} kW (~{final_physics_df['DRS_Power_Gain_kW'].max() * 1.35962:.1f} HP)")
    
    print("\nSample Output in DRS Zone:")
    active_drs_sample = final_physics_df[final_physics_df['DRS_Active']].head(5)
    print(active_drs_sample[['Distance', 'Speed', 'DRS', 'Delta_Fx_Drag_N', 'DRS_Power_Gain_kW']])