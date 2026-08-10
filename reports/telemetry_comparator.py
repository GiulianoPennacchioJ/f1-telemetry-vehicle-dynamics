import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class TelemetryComparator:
    """
    Compares two spatial telemetry datasets on a unified distance grid,
    computing instantaneous delta metrics (Δv, Δt) and physical channel differences.
    """

    def __init__(self, driver_ref_name: str = "REF", driver_comp_name: str = "COMP"):
        """
        Parameters:
        -----------
        driver_ref_name : str
            Identifier for the reference driver/lap (e.g., 'NOR').
        driver_comp_name : str
            Identifier for the comparison driver/lap (e.g., 'VER').
        """
        self.ref_name = driver_ref_name
        self.comp_name = driver_comp_name

    def compute_spatial_delta_time(self, v_ref_ms: np.ndarray, v_comp_ms: np.ndarray, ds: float = 1.0) -> np.ndarray:
        """
        Calculates accumulated delta time along the spatial track grid:
        Δt(s) = ∫ (1 / v_comp - 1 / v_ref) ds
        
        A positive value means Driver Ref is FASTER than Driver Comp.
        A negative value means Driver Ref is SLOWER than Driver Comp.
        """
        # Guard against division by zero at low speeds
        safe_v_ref = np.maximum(v_ref_ms, 1.0)
        safe_v_comp = np.maximum(v_comp_ms, 1.0)

        # Differential time element per spatial step dt = ds / v
        dt_ref = ds / safe_v_ref
        dt_comp = ds / safe_v_comp

        # Time difference per step and cumulative integration
        dt_diff = dt_comp - dt_ref
        delta_time_accumulated = np.cumsum(dt_diff)

        return delta_time_accumulated

    def compare_laps(self, df_ref: pd.DataFrame, df_comp: pd.DataFrame) -> pd.DataFrame:
        """
        Merges and aligns two processed driver DataFrames to compute comparative metrics.

        Parameters:
        -----------
        df_ref : pd.DataFrame
            Processed telemetry DataFrame for reference driver.
        df_comp : pd.DataFrame
            Processed telemetry DataFrame for comparison driver.

        Returns:
        --------
        comparison_df : pd.DataFrame
            Merged spatial DataFrame with delta time, delta speed, and driver-specific channels.
        """
        # Ensure distance grid consistency
        min_len = min(len(df_ref), len(df_comp))
        ref = df_ref.iloc[:min_len].copy()
        comp = df_comp.iloc[:min_len].copy()

        # Extract speeds in m/s
        v_ref_kmh = ref['Speed_Smoothed_kmh'].values if 'Speed_Smoothed_kmh' in ref.columns else ref['Speed'].values
        v_comp_kmh = comp['Speed_Smoothed_kmh'].values if 'Speed_Smoothed_kmh' in comp.columns else comp['Speed'].values

        v_ref_ms = v_ref_kmh / 3.6
        v_comp_ms = v_comp_kmh / 3.6

        # Determine spatial step size ds (assumed constant from spatial resampler)
        ds = float(np.mean(np.diff(ref['Distance'].values))) if len(ref) > 1 else 1.0

        # Compute delta metrics
        delta_t = self.compute_spatial_delta_time(v_ref_ms, v_comp_ms, ds=ds)
        delta_v_kmh = v_ref_kmh - v_comp_kmh

        # Build combined comparative DataFrame
        comp_df = pd.DataFrame({
            'Distance': ref['Distance'].values,
            'X': ref.get('X', np.nan),  # <-- Aggiunto passaggio coordinata X
            'Y': ref.get('Y', np.nan),  # <-- Aggiunto passaggio coordinata Y
            f'Speed_{self.ref_name}': v_ref_kmh,
            f'Speed_{self.comp_name}': v_comp_kmh,
            'Delta_Speed_kmh': delta_v_kmh,
            'Delta_Time_sec': delta_t,
            f'Throttle_{self.ref_name}': ref.get('Throttle', np.nan),
            f'Throttle_{self.comp_name}': comp.get('Throttle', np.nan),
            f'Brake_{self.ref_name}': ref.get('Brake', np.nan),
            f'Brake_{self.comp_name}': comp.get('Brake', np.nan),
            f'Fz_Total_{self.ref_name}': ref.get('Fz_Total_N', np.nan),
            f'Fz_Total_{self.comp_name}': comp.get('Fz_Total_N', np.nan),
            f'Mu_Util_{self.ref_name}': ref.get('Mu_Utilized', np.nan),
            f'Mu_Util_{self.comp_name}': comp.get('Mu_Utilized', np.nan),
            f'Power_Wheels_{self.ref_name}': ref.get('Power_Wheels_kW', np.nan),
            f'Power_Wheels_{self.comp_name}': comp.get('Power_Wheels_kW', np.nan)
        })

        logging.info(f"Lap comparison between {self.ref_name} and {self.comp_name} completed successfully.")
        return comp_df


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
    from physics.drs_analyzer import DRSAnalyzer

    def process_driver_pipeline(session, driver_code: str):
        loader = F1DataLoader()
        lap = loader.get_driver_fastest_lap(session, driver_code)
        raw_df = loader.extract_raw_telemetry(lap)
        
        resampler = SpatialResampler(step_size_meters=1.0)
        spatial_df = resampler.resample_lap(raw_df)

        processor = SignalProcessor(window_length=15, poly_order=3)
        kinematic_df = processor.compute_kinematics(spatial_df)

        aero = AeroModel(config_path="config/car_parameters.json")
        aero_df = aero.process_telemetry_aero(kinematic_df)

        tire = TireModel(config_path="config/car_parameters.json")
        tire_df = tire.process_telemetry_tires(aero_df)

        gg = GGDiagramAnalyzer()
        gg_df, _ = gg.process_telemetry_gg(tire_df)

        ers = ERSAnalyzer(config_path="config/car_parameters.json")
        ers_df = ers.process_telemetry_ers(gg_df)

        drs = DRSAnalyzer(config_path="config/car_parameters.json")
        final_df, _ = drs.process_telemetry_drs(ers_df)
        
        return final_df

    # Load and process NOR vs VER (Monza 2025 Q)
    loader = F1DataLoader()
    session = loader.load_session(2025, 'Monza', 'Q')

    print("Processing NOR pipeline...")
    df_nor = process_driver_pipeline(session, 'NOR')
    print("Processing VER pipeline...")
    df_ver = process_driver_pipeline(session, 'VER')

    # Compare Norris vs Verstappen
    comparator = TelemetryComparator(driver_ref_name="NOR", driver_comp_name="VER")
    comp_df = comparator.compare_laps(df_nor, df_ver)

    print("\n==========================================")
    print(" TELEMETRY COMPARISON SUCCESSFUL")
    print("==========================================")
    print(f"Final Delta Time (NOR vs VER): {comp_df['Delta_Time_sec'].iloc[-1]:+.3f} seconds")
    print("Max Speed Delta (NOR - VER):   ", f"{comp_df['Delta_Speed_kmh'].max():+.1f} km/h")
    print("Min Speed Delta (NOR - VER):   ", f"{comp_df['Delta_Speed_kmh'].min():+.1f} km/h")
    
    print("\nSample Comparative Telemetry Output:")
    print(comp_df[['Distance', 'Speed_NOR', 'Speed_VER', 'Delta_Speed_kmh', 'Delta_Time_sec']].head(10))