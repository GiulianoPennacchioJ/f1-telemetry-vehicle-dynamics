import os
import logging
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class GGDiagramAnalyzer:
    """
    Analyzes the G-G acceleration envelope (friction circle) and classifies 
    driving dynamics domains across the spatial telemetry grid.
    """

    def __init__(self, g_const: float = 9.81):
        """
        Parameters:
        -----------
        g_const : float
            Gravitational acceleration constant [m/s^2].
        """
        self.g = g_const

    def compute_g_forces(self, ax_ms2: np.ndarray, ay_ms2: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Converts acceleration from m/s^2 to g-units and calculates scalar magnitude:
        a_tot_g = sqrt(a_x_g^2 + a_y_g^2)
        """
        ax_g = ax_ms2 / self.g
        ay_g = ay_ms2 / self.g
        a_tot_g = np.sqrt(ax_g**2 + ay_g**2)
        return ax_g, ay_g, a_tot_g

    def classify_driving_regime(self, ax_g: np.ndarray, ay_g: np.ndarray) -> np.ndarray:
        """
        Categorizes each spatial point into a vehicle dynamics regime.
        """
        conditions = [
            (ax_g < -0.5) & (np.abs(ay_g) < 0.5),   # Straight-line braking
            (ax_g < -0.2) & (np.abs(ay_g) >= 0.5),  # Combined braking & cornering (trail braking)
            (np.abs(ax_g) <= 0.2) & (np.abs(ay_g) >= 0.5), # Pure lateral cornering
            (ax_g > 0.2)                             # Longitudinal acceleration / traction
        ]
        choices = ['Straight Braking', 'Trail Braking', 'Pure Cornering', 'Traction']
        return np.select(conditions, choices, default='Coasting / Straight')

    def extract_convex_hull_boundary(self, ax_g: np.ndarray, ay_g: np.ndarray) -> pd.DataFrame:
        """
        Extracts the outer boundary envelope of the G-G diagram using 2D Convex Hull.
        """
        points = np.column_stack((ay_g, ax_g)) # ay on X-axis, ax on Y-axis
        
        # Guard against degenerate points set
        if len(points) < 3:
            return pd.DataFrame(columns=['a_y_g_hull', 'a_x_g_hull'])

        hull = ConvexHull(points)
        hull_points = points[hull.vertices]
        
        # Close the loop for plotting continuity
        hull_points = np.vstack([hull_points, hull_points[0]])
        
        return pd.DataFrame({
            'a_y_g_hull': hull_points[:, 0],
            'a_x_g_hull': hull_points[:, 1]
        })

    def process_telemetry_gg(self, tire_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Processes acceleration data to construct the G-G performance envelope.

        Parameters:
        -----------
        tire_df : pd.DataFrame
            Telemetry DataFrame containing kinematic accelerations.

        Returns:
        --------
        gg_df : pd.DataFrame
            DataFrame augmented with 'a_x_g', 'a_y_g', 'a_tot_g', and 'Driving_Regime'.
        hull_df : pd.DataFrame
            Convex Hull boundary points for visualization.
        """
        df = tire_df.copy()

        # Extract accelerations in m/s^2
        ax_ms2 = df['a_x'].values if 'a_x' in df.columns else df.get('a_x_ms2', np.zeros(len(df))).values
        ay_ms2 = df['a_y'].values if 'a_y' in df.columns else df.get('a_y_ms2', np.zeros(len(df))).values

        # Compute G-forces and regimes
        ax_g, ay_g, a_tot_g = self.compute_g_forces(ax_ms2, ay_ms2)
        regimes = self.classify_driving_regime(ax_g, ay_g)

        # Assign output channels
        df['a_x_g'] = ax_g
        df['a_y_g'] = ay_g
        df['a_tot_g'] = a_tot_g
        df['Driving_Regime'] = regimes

        # Extract outer boundary envelope
        hull_df = self.extract_convex_hull_boundary(ax_g, ay_g)

        logging.info("G-G diagram envelope and dynamic driving regimes processed successfully.")
        return df, hull_df


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

    # Execution pipeline up to TireModel
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

    # Instantiate and execute GGDiagramAnalyzer
    gg_analyzer = GGDiagramAnalyzer()
    gg_df, hull_df = gg_analyzer.process_telemetry_gg(tire_df)

    print("\n==========================================")
    print(" G-G DIAGRAM ANALYSIS SUCCESSFUL")
    print("==========================================")
    print("Max Longitudinal Acceleration (a_x):", f"{gg_df['a_x_g'].max():.2f} G")
    print("Max Longitudinal Braking (a_x):     ", f"{gg_df['a_x_g'].min():.2f} G")
    print("Max Lateral Cornering (a_y):        ", f"{np.abs(gg_df['a_y_g']).max():.2f} G")
    print("\nDriving Regimes Breakdown:")
    print(gg_df['Driving_Regime'].value_counts())