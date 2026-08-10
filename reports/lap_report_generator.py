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
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

from reports.telemetry_comparator import TelemetryComparator
from reports.efficiency_index import EfficiencyIndexAnalyzer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class LapReportGenerator:
    """
    Synthesizes comparison telemetry, physics KPIs, and spatial telemetry
    into visual executive plots (Telemetry Overlay, Track Dominance Map, GG Diagram, ERS Clipping, Aero Forces).
    """

    def __init__(self, output_dir: str = "reports/output"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_comparison_plot(self, comp_df: pd.DataFrame, ref_name: str = "NOR", comp_name: str = "VER", save_filename: str = "lap_comparison.png") -> str:
        fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1, 1, 1]})
        
        dist_ref = comp_df['Distance'].values if 'Distance' in comp_df.columns else np.arange(len(comp_df))

        # Panel 1: Speed Overlay
        axes[0].plot(dist_ref, comp_df[f'Speed_{ref_name}'], label=f'{ref_name}', color='orange', linewidth=1.5)
        axes[0].plot(dist_ref, comp_df[f'Speed_{comp_name}'], label=f'{comp_name}', color='blue', linewidth=1.5, linestyle='--')
        axes[0].set_ylabel("Speed [km/h]")
        axes[0].set_title(f"Executive Lap Comparison: {ref_name} vs {comp_name} (Monza 2025)", fontsize=12, fontweight='bold')
        axes[0].grid(True, linestyle=':', alpha=0.6)
        axes[0].legend(loc='upper right')

        # Panel 2: Accumulated Delta Time
        axes[1].plot(dist_ref, comp_df['Delta_Time_sec'], color='purple', linewidth=1.5)
        axes[1].axhline(0, color='black', linestyle='--', alpha=0.5)
        axes[1].set_ylabel("Δt Accum. [s]")
        axes[1].grid(True, linestyle=':', alpha=0.6)

        # Panel 3: Throttle Position
        axes[2].plot(dist_ref, comp_df[f'Throttle_{ref_name}'], label=f'Throttle {ref_name}', color='orange', alpha=0.8)
        axes[2].plot(dist_ref, comp_df[f'Throttle_{comp_name}'], label=f'Throttle {comp_name}', color='blue', alpha=0.6, linestyle='--')
        axes[2].set_ylabel("Throttle [%]")
        axes[2].grid(True, linestyle=':', alpha=0.6)
        axes[2].legend(loc='lower right')

        # Panel 4: Power at Wheels
        p_col_ref = f'Power_Wheels_{ref_name}' if f'Power_Wheels_{ref_name}' in comp_df.columns else f'P_wheels_kW_{ref_name}'
        p_col_comp = f'Power_Wheels_{comp_name}' if f'Power_Wheels_{comp_name}' in comp_df.columns else f'P_wheels_kW_{comp_name}'
        
        p_ref_vals = comp_df[p_col_ref].values if p_col_ref in comp_df.columns else np.zeros(len(comp_df))
        p_comp_vals = comp_df[p_col_comp].values if p_col_comp in comp_df.columns else np.zeros(len(comp_df))

        axes[3].plot(dist_ref, p_ref_vals, label=f'P_wheels {ref_name}', color='orange', alpha=0.8)
        axes[3].plot(dist_ref, p_comp_vals, label=f'P_wheels {comp_name}', color='blue', alpha=0.6, linestyle='--')
        axes[3].set_ylabel("Wheel Power [kW]")
        axes[3].set_xlabel("Track Distance [m]")
        axes[3].grid(True, linestyle=':', alpha=0.6)
        axes[3].legend(loc='lower right')

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, save_filename)
        plt.savefig(save_path, dpi=300)
        plt.close()
        logging.info(f"Comparison plot saved to: {save_path}")
        return save_path

    def generate_track_dominance_map(self, comp_df: pd.DataFrame, ref_name: str = "NOR", comp_name: str = "VER", save_filename: str = "track_dominance_map.png") -> str:
        if 'X' not in comp_df.columns or 'Y' not in comp_df.columns:
            logging.warning("GPS coordinates (X, Y) not found in comp_df. Skipping Track Dominance Map generation.")
            return ""

        x = comp_df['X'].values
        y = comp_df['Y'].values

        delta_v = comp_df['Delta_Speed_kmh'].values if 'Delta_Speed_kmh' in comp_df.columns else (comp_df[f'Speed_{ref_name}'].values - comp_df[f'Speed_{comp_name}'].values)

        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        fig, ax = plt.subplots(figsize=(10, 8))
        cmap = ListedColormap(['#0055ff', '#cccccc', '#ff8700'])
        norm = BoundaryNorm([-30.0, -1.5, 1.5, 30.0], cmap.N)

        lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=4)
        lc.set_array(delta_v)

        line = ax.add_collection(lc)
        cbar = fig.colorbar(line, ax=ax, ticks=[-15, 0, 15], orientation='horizontal', pad=0.05, shrink=0.7)
        cbar.ax.set_xticklabels([f'{comp_name} Faster (>1.5 km/h)', 'Equal Speed (±1.5 km/h)', f'{ref_name} Faster (>1.5 km/h)'])

        ax.set_title(f"Track Speed Dominance Map: {ref_name} vs {comp_name}", fontsize=12, fontweight='bold')
        ax.axis('equal')
        ax.axis('off')

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, save_filename)
        plt.savefig(save_path, dpi=300)
        plt.close()
        logging.info(f"Track dominance map saved to: {save_path}")
        return save_path

    def generate_gg_diagram_plot(self, df_ref: pd.DataFrame, df_comp: pd.DataFrame, ref_name: str = "NOR", comp_name: str = "VER", save_filename: str = "gg_diagram_comparison.png") -> str:
        fig, ax = plt.subplots(figsize=(8, 8))

        ax_ref = df_ref['a_x'].values if 'a_x' in df_ref.columns else np.zeros(len(df_ref))
        ay_ref = df_ref['a_y'].values if 'a_y' in df_ref.columns else np.zeros(len(df_ref))
        ax_comp = df_comp['a_x'].values if 'a_x' in df_comp.columns else np.zeros(len(df_comp))
        ay_comp = df_comp['a_y'].values if 'a_y' in df_comp.columns else np.zeros(len(df_comp))

        ax.scatter(ay_ref, ax_ref, color='orange', alpha=0.3, s=10, label=f'{ref_name}')
        ax.scatter(ay_comp, ax_comp, color='blue', alpha=0.2, s=10, label=f'{comp_name}')

        ax.axhline(0, color='black', linestyle='--', alpha=0.6)
        ax.axvline(0, color='black', linestyle='--', alpha=0.6)
        ax.set_xlabel("Lateral Acceleration a_y [G]")
        ax.set_ylabel("Longitudinal Acceleration a_x [G]")
        ax.set_title(f"GG Diagram Friction Utilization: {ref_name} vs {comp_name}", fontsize=12, fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='upper right')
        ax.set_xlim(-6.0, 6.0)
        ax.set_ylim(-6.0, 3.0)

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, save_filename)
        plt.savefig(save_path, dpi=300)
        plt.close()
        logging.info(f"GG Diagram plot saved to: {save_path}")
        return save_path

    def generate_ers_clipping_plot(self, df_ref: pd.DataFrame, df_comp: pd.DataFrame, ref_name: str = "NOR", comp_name: str = "VER", save_filename: str = "ers_clipping_analysis.png") -> str:
        fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
        
        dist_ref = df_ref['Distance'].values if 'Distance' in df_ref.columns else np.arange(len(df_ref))
        dist_comp = df_comp['Distance'].values if 'Distance' in df_comp.columns else np.arange(len(df_comp))

        # Estrazione corretta da ERSAnalyzer (Power_Wheels_kW)
        p_ref = df_ref['Power_Wheels_kW'].values if 'Power_Wheels_kW' in df_ref.columns else df_ref.get('Power_Wheels', np.zeros(len(df_ref))).values
        p_comp = df_comp['Power_Wheels_kW'].values if 'Power_Wheels_kW' in df_comp.columns else df_comp.get('Power_Wheels', np.zeros(len(df_comp))).values

        clip_ref = df_ref['Is_Clipping'].values if 'Is_Clipping' in df_ref.columns else np.zeros(len(df_ref))
        clip_comp = df_comp['Is_Clipping'].values if 'Is_Clipping' in df_comp.columns else np.zeros(len(df_comp))

        # Panel 1: Power at Wheels
        axes[0].plot(dist_ref, p_ref, label=f'{ref_name}', color='orange', linewidth=1.5)
        axes[0].plot(dist_comp, p_comp, label=f'{comp_name}', color='blue', linewidth=1.5, linestyle='--')
        axes[0].set_ylabel("Power at Wheels [kW]")
        axes[0].set_title(f"ERS Deployment & Clipping Analysis: {ref_name} vs {comp_name}", fontsize=12, fontweight='bold')
        axes[0].grid(True, linestyle=':', alpha=0.6)
        axes[0].legend(loc='upper right')

        # Panel 2: Clipping State Flag
        axes[1].fill_between(dist_ref, 0, clip_ref.astype(int), color='orange', alpha=0.5, label=f'Clipping {ref_name}')
        axes[1].fill_between(dist_comp, 0, clip_comp.astype(int), color='blue', alpha=0.3, label=f'Clipping {comp_name}')
        axes[1].set_ylabel("Clipping State")
        axes[1].set_xlabel("Track Distance [m]")
        axes[1].set_yticks([0, 1])
        axes[1].set_yticklabels(['Deploying', 'Clipping'])
        axes[1].grid(True, linestyle=':', alpha=0.6)
        axes[1].legend(loc='upper right')

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, save_filename)
        plt.savefig(save_path, dpi=300)
        plt.close()
        logging.info(f"ERS Clipping plot saved to: {save_path}")
        return save_path

    def generate_aero_forces_plot(self, df_ref: pd.DataFrame, df_comp: pd.DataFrame, ref_name: str = "NOR", comp_name: str = "VER", save_filename: str = "aero_forces_comparison.png") -> str:
        fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
        
        dist_ref = df_ref['Distance'].values if 'Distance' in df_ref.columns else np.arange(len(df_ref))
        dist_comp = df_comp['Distance'].values if 'Distance' in df_comp.columns else np.arange(len(df_comp))

        fz_ref = df_ref['Fz_Downforce_N'].values if 'Fz_Downforce_N' in df_ref.columns else np.zeros(len(df_ref))
        fz_comp = df_comp['Fz_Downforce_N'].values if 'Fz_Downforce_N' in df_comp.columns else np.zeros(len(df_comp))

        fx_ref = df_ref['Fx_Drag_N'].values if 'Fx_Drag_N' in df_ref.columns else np.zeros(len(df_ref))
        fx_comp = df_comp['Fx_Drag_N'].values if 'Fx_Drag_N' in df_comp.columns else np.zeros(len(df_comp))

        # Panel 1: Downforce
        axes[0].plot(dist_ref, fz_ref, label=f'{ref_name}', color='orange', linewidth=1.5)
        axes[0].plot(dist_comp, fz_comp, label=f'{comp_name}', color='blue', linewidth=1.5, linestyle='--')
        axes[0].set_ylabel("Downforce Fz [N]")
        axes[0].set_title(f"Aerodynamic Loads Comparison: {ref_name} vs {comp_name}", fontsize=12, fontweight='bold')
        axes[0].grid(True, linestyle=':', alpha=0.6)
        axes[0].legend(loc='upper right')

        # Panel 2: Drag
        axes[1].plot(dist_ref, fx_ref, label=f'{ref_name}', color='orange', linewidth=1.5)
        axes[1].plot(dist_comp, fx_comp, label=f'{comp_name}', color='blue', linewidth=1.5, linestyle='--')
        axes[1].set_ylabel("Aero Drag Fx [N]")
        axes[1].set_xlabel("Track Distance [m]")
        axes[1].grid(True, linestyle=':', alpha=0.6)
        axes[1].legend(loc='upper right')

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, save_filename)
        plt.savefig(save_path, dpi=300)
        plt.close()
        logging.info(f"Aero forces plot saved to: {save_path}")
        return save_path

    def build_executive_summary(self, comp_df: pd.DataFrame, kpi_ref: dict, kpi_comp: dict) -> pd.DataFrame:
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

    comparator = TelemetryComparator(driver_ref_name="NOR", driver_comp_name="VER")
    comp_df = comparator.compare_laps(df_nor, df_ver)

    efficiency = EfficiencyIndexAnalyzer(ds=1.0)
    kpi_nor = efficiency.analyze_lap_efficiency(df_nor, driver_name="NOR")
    kpi_ver = efficiency.analyze_lap_efficiency(df_ver, driver_name="VER")

    reporter = LapReportGenerator()
    p1 = reporter.generate_comparison_plot(comp_df, ref_name="NOR", comp_name="VER")
    p2 = reporter.generate_track_dominance_map(comp_df, ref_name="NOR", comp_name="VER")
    p3 = reporter.generate_gg_diagram_plot(df_nor, df_ver, ref_name="NOR", comp_name="VER")
    p4 = reporter.generate_ers_clipping_plot(df_nor, df_ver, ref_name="NOR", comp_name="VER")
    p5 = reporter.generate_aero_forces_plot(df_nor, df_ver, ref_name="NOR", comp_name="VER")
    
    summary_table = reporter.build_executive_summary(comp_df, kpi_nor, kpi_ver)

    print("\n==========================================")
    print(" ALL 5 EXECUTIVE REPORTS GENERATED")
    print("==========================================")
    print(f"1. Telemetry Overlay Plot: {p1}")
    print(f"2. Track Dominance Map:    {p2}")
    print(f"3. GG Diagram Plot:       {p3}")
    print(f"4. ERS Clipping Plot:     {p4}")
    print(f"5. Aero Forces Plot:       {p5}")
    print("\nExecutive Summary Table:")
    print(summary_table.to_string(index=False))