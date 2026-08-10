import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class EfficiencyIndexAnalyzer:
    """
    Computes global performance indicators (KPIs) for a completed lap,
    including Integrated Aero Efficiency (L/D), ERS Clipping Ratio, and Mean Grip Utilization.
    """

    def __init__(self, ds: float = 1.0):
        """
        Parameters:
        -----------
        ds : float
            Spatial discretization step in meters.
        """
        self.ds = ds

    def compute_aero_efficiency(self, fz_downforce: np.ndarray, fx_drag: np.ndarray) -> float:
        """
        Calculates spatially integrated Aerodynamic Efficiency Ratio:
        Integrated (L/D) = ∫ Fz_aero ds / ∫ Fx_drag ds
        """
        total_downforce_work = np.sum(fz_downforce) * self.ds
        total_drag_work = np.sum(fx_drag) * self.ds

        if total_drag_work <= 0:
            return 0.0

        return float(total_downforce_work / total_drag_work)

    def compute_clipping_ratio(self, is_clipping: np.ndarray) -> tuple[float, float]:
        """
        Calculates ERS Clipping Distance [m] and Clipping Ratio [%] over total lap distance.
        """
        clipping_points = np.sum(is_clipping)
        clipping_distance_m = clipping_points * self.ds
        total_distance_m = len(is_clipping) * self.ds

        clipping_ratio_pct = (clipping_distance_m / total_distance_m) * 100.0 if total_distance_m > 0 else 0.0

        return float(clipping_distance_m), float(clipping_ratio_pct)

    def compute_mean_grip_utilization(self, mu_utilized: np.ndarray) -> float:
        """
        Calculates average friction coefficient utilization across the entire lap.
        """
        valid_mu = mu_utilized[~np.isnan(mu_utilized)]
        if len(valid_mu) == 0:
            return 0.0

        return float(np.mean(valid_mu))

    def analyze_lap_efficiency(self, df_processed: pd.DataFrame, driver_name: str = "DRIVER") -> dict:
        """
        Synthesizes telemetry channels into an executive efficiency dictionary.

        Parameters:
        -----------
        df_processed : pd.DataFrame
            Complete processed telemetry DataFrame from physics pipeline.
        driver_name : str
            Driver code identifier.

        Returns:
        --------
        kpi_metrics : dict
            Dictionary of calculated performance efficiency metrics.
        """
        # 1. Aero Downforce vs Drag extraction
        if 'Fz_Downforce_N' in df_processed.columns:
            fz_aero = df_processed['Fz_Downforce_N'].values
        elif 'Fz_Aero_N' in df_processed.columns:
            fz_aero = df_processed['Fz_Aero_N'].values
        else:
            fz_aero = np.zeros(len(df_processed))

        if 'Fx_Drag_N' in df_processed.columns:
            fx_drag = df_processed['Fx_Drag_N'].values
        elif 'Fx_Drag' in df_processed.columns:
            fx_drag = df_processed['Fx_Drag'].values
        else:
            fx_drag = np.ones(len(df_processed))

        # 2. ERS Clipping state extraction
        if 'Is_Clipping' in df_processed.columns:
            is_clipping = df_processed['Is_Clipping'].values
        else:
            is_clipping = np.zeros(len(df_processed), dtype=bool)

        # 3. Corrected Friction Utilization (Mu) extraction
        # If Mu_Utilized in df was computed using G's instead of m/s^2, re-scale by g (9.81)
        if 'Mu_Utilized' in df_processed.columns:
            mu_raw = df_processed['Mu_Utilized'].values
            # Check if values are unscaled (mean < 0.1) and apply gravity scaling
            if np.nanmean(mu_raw) < 0.1 and np.nanmean(mu_raw) > 0:
                mu_util = mu_raw * 9.81
            else:
                mu_util = mu_raw
        else:
            mu_util = np.zeros(len(df_processed))

        aero_efficiency = self.compute_aero_efficiency(fz_aero, fx_drag)
        clip_dist_m, clip_ratio_pct = self.compute_clipping_ratio(is_clipping)
        mean_mu = self.compute_mean_grip_utilization(mu_util)

        kpi_summary = {
            'Driver': driver_name,
            'Aero_Efficiency_L_D': round(aero_efficiency, 3),
            'Clipping_Distance_m': round(clip_dist_m, 1),
            'Clipping_Ratio_Pct': round(clip_ratio_pct, 2),
            'Mean_Grip_Utilization': round(mean_mu, 3)
        }

        logging.info(f"Efficiency KPIs calculated successfully for driver: {driver_name}")
        return kpi_summary


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

    loader = F1DataLoader()
    session = loader.load_session(2025, 'Monza', 'Q')

    df_nor = process_driver_pipeline(session, 'NOR')
    df_ver = process_driver_pipeline(session, 'VER')

    analyzer = EfficiencyIndexAnalyzer(ds=1.0)
    kpi_nor = analyzer.analyze_lap_efficiency(df_nor, driver_name="NOR")
    kpi_ver = analyzer.analyze_lap_efficiency(df_ver, driver_name="VER")

    summary_df = pd.DataFrame([kpi_nor, kpi_ver])

    print("\n==========================================")
    print(" EFFICIENCY KPI ANALYSIS SUCCESSFUL")
    print("==========================================")
    print(summary_df.to_string(index=False))