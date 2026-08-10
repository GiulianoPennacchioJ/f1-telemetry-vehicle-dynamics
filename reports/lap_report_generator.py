import os
import sys
import logging
from pathlib import Path

# Add project root directory to sys.path before importing internal packages
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from reports.telemetry_comparator import TelemetryComparator
from reports.efficiency_index import EfficiencyIndexAnalyzer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class LapReportGenerator:
    """
    Synthesizes comparison telemetry and performance KPIs into executive visual reports and summary plots.
    """

    def __init__(self, output_dir: str = "reports/output"):
        """
        Parameters:
        -----------
        output_dir : str
            Directory path to save generated plots and reports.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_comparison_plot(self, comp_df: pd.DataFrame, ref_name: str = "NOR", comp_name: str = "VER", save_filename: str = "lap_comparison.png") -> str:
        """
        Generates a multi-panel spatial telemetry trace overlay plot.
        """
        fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1, 1, 1]})

        distance = comp_df['Distance'].values

        # Panel 1: Speed Overlay
        axes[0].plot(distance, comp_df[f'Speed_{ref_name}'], label=f'{ref_name}', color='orange', linewidth=1.5)
        axes[0].plot(distance, comp_df[f'Speed_{comp_name}'], label=f'{comp_name}', color='blue', linewidth=1.5, linestyle='--')
        axes[0].set_ylabel("Speed [km/h]")
        axes[0].set_title(f"Executive Lap Comparison: {ref_name} vs {comp_name} (Monza 2025)", fontsize=12, fontweight='bold')
        axes[0].grid(True, linestyle=':', alpha=0.6)
        axes[0].legend(loc='upper right')

        # Panel 2: Accumulated Delta Time
        axes[1].plot(distance, comp_df['Delta_Time_sec'], color='purple', linewidth=1.5)
        axes[1].axhline(0, color='black', linestyle='--', alpha=0.5)
        axes[1].set_ylabel("Δt Accum. [s]")
        axes[1].grid(True, linestyle=':', alpha=0.6)

        # Panel 3: Pedal Inputs (Throttle)
        axes[2].plot(distance, comp_df[f'Throttle_{ref_name}'], label=f'Throttle {ref_name}', color='orange', alpha=0.8)
        axes[2].plot(distance, comp_df[f'Throttle_{comp_name}'], label=f'Throttle {comp_name}', color='blue', alpha=0.6, linestyle='--')
        axes[2].set_ylabel("Throttle [%]")
        axes[2].grid(True, linestyle=':', alpha=0.6)
        axes[2].legend(loc='lower right')

        # Panel 4: Power at Wheels
        axes[3].plot(distance, comp_df[f'Power_Wheels_{ref_name}'], label=f'P_wheels {ref_name}', color='orange', alpha=0.8)
        axes[3].plot(distance, comp_df[f'Power_Wheels_{comp_name}'], label=f'P_wheels {comp_name}', color='blue', alpha=0.6, linestyle='--')
        axes[3].set_ylabel("Wheel Power [kW]")
        axes[3].set_xlabel("Track Distance [m]")
        axes[3].grid(True, linestyle=':', alpha=0.6)
        axes[3].legend(loc='lower right')

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, save_filename)
        plt.savefig(save_path, dpi=300)
        plt.close()

        logging.info(f"Comparison plot saved successfully to: {save_path}")
        return save_path

    def build_executive_summary(self, comp_df: pd.DataFrame, kpi_ref: dict, kpi_comp: dict) -> pd.DataFrame:
        """
        Builds a structured executive summary table comparing two drivers.
        """
        ref_name = kpi_ref['Driver']
        comp_name = kpi_comp['Driver']

        delta_final_time = comp_df['Delta_Time_sec'].iloc[-1]
        max_speed_ref = comp_df[f'Speed_{ref_name}'].max()
        max_speed_comp = comp_df[f'Speed_{comp_name}'].max()

        summary_data = {
            'Metric': [
                'Final Lap Delta [s]',
                'Max Speed [km/h]',
                'Integrated Aero Efficiency (L/D)',
                'ERS Clipping Distance [m]',
                'ERS Clipping Ratio [%]',
                'Mean Grip Utilization'
            ],
            ref_name: [
                "0.000 (REF)",
                f"{max_speed_ref:.1f}",
                f"{kpi_ref['Aero_Efficiency_L_D']:.3f}",
                f"{kpi_ref['Clipping_Distance_m']:.1f}",
                f"{kpi_ref['Clipping_Ratio_Pct']:.2f}%",
                f"{kpi_ref['Mean_Grip_Utilization']:.3f}"
            ],
            comp_name: [
                f"{delta_final_time:+.3f}",
                f"{max_speed_comp:.1f}",
                f"{kpi_comp['Aero_Efficiency_L_D']:.3f}",
                f"{kpi_comp['Clipping_Distance_m']:.1f}",
                f"{kpi_comp['Clipping_Ratio_Pct']:.2f}%",
                f"{kpi_comp['Mean_Grip_Utilization']:.3f}"
            ]
        }

        return pd.DataFrame(summary_data)


if __name__ == "__main__":
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

    # Execute comparison and KPI pipeline
    comparator = TelemetryComparator(driver_ref_name="NOR", driver_comp_name="VER")
    comp_df = comparator.compare_laps(df_nor, df_ver)

    efficiency = EfficiencyIndexAnalyzer(ds=1.0)
    kpi_nor = efficiency.analyze_lap_efficiency(df_nor, driver_name="NOR")
    kpi_ver = efficiency.analyze_lap_efficiency(df_ver, driver_name="VER")

    # Generate visual report and summary table
    reporter = LapReportGenerator()
    plot_file = reporter.generate_comparison_plot(comp_df, ref_name="NOR", comp_name="VER")
    summary_table = reporter.build_executive_summary(comp_df, kpi_nor, kpi_ver)

    print("\n==========================================")
    print(" EXECUTIVE LAP REPORT GENERATED")
    print("==========================================")
    print(f"Plot saved at: {plot_file}")
    print("\nExecutive Summary Table:")
    print(summary_table.to_string(index=False))