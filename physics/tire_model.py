import logging
import numpy as np
import pandas as pd
import json

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class TireModel:
    """
    Computes tire normal loads, combined forces, and friction utilization (Mu_Utilized).
    """

    def __init__(self, config_path: str = "config/car_parameters.json"):
        with open(config_path, 'r') as f:
            self.params = json.load(f)

        self.mass = float(self.params.get('mass_kg', 798.0))
        self.g = 9.81  # m/s^2

    def compute_grip_utilization(self, ax_m_s2: np.ndarray, ay_m_s2: np.ndarray, fz_aero_n: np.ndarray) -> np.ndarray:
        """
        Calculates dimensionless friction coefficient utilization:
        Mu_Utilized = Total_Planar_Force [N] / Total_Normal_Force [N]
        
        Parameters:
        -----------
        ax_m_s2 : np.ndarray
            Longitudinal acceleration in m/s^2.
        ay_m_s2 : np.ndarray
            Lateral acceleration in m/s^2.
        fz_aero_n : np.ndarray
            Aerodynamic downforce in Newtons.
        """
        # 1. Force in the contact plane (F_xy = m * a_total) [N]
        a_total_m_s2 = np.sqrt(ax_m_s2**2 + ay_m_s2**2)
        f_xy_total = self.mass * a_total_m_s2

        # 2. Total normal load (F_z_total = m * g + F_z_aero) [N]
        f_z_static = self.mass * self.g
        f_z_total = f_z_static + fz_aero_n

        # Avoid division by zero
        f_z_total = np.maximum(f_z_total, 1.0)

        # 3. Dimensionless Grip Utilization (Mu)
        mu_utilized = f_xy_total / f_z_total

        return mu_utilized

    def process_telemetry_tires(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Processes DataFrame to extract accelerations, compute normal forces and grip utilization.
        """
        df_out = df.copy()

        # Extract accelerations, ensure they are in m/s^2
        if 'ax_m_s2' in df_out.columns:
            ax = df_out['ax_m_s2'].values
        elif 'ax_g' in df_out.columns:
            ax = df_out['ax_g'].values * self.g
        elif 'ax' in df_out.columns:
            ax = df_out['ax'].values * self.g if np.max(np.abs(df_out['ax'])) < 10.0 else df_out['ax'].values
        else:
            ax = np.zeros(len(df_out))

        if 'ay_m_s2' in df_out.columns:
            ay = df_out['ay_m_s2'].values
        elif 'ay_g' in df_out.columns:
            ay = df_out['ay_g'].values * self.g
        elif 'ay' in df_out.columns:
            ay = df_out['ay'].values * self.g if np.max(np.abs(df_out['ay'])) < 10.0 else df_out['ay'].values
        else:
            ay = np.zeros(len(df_out))

        # Extract Aero Downforce [N]
        if 'Fz_Aero_N' in df_out.columns:
            fz_aero = df_out['Fz_Aero_N'].values
        elif 'Fz_Aero' in df_out.columns:
            fz_aero = df_out['Fz_Aero'].values
        elif 'Fz_Downforce_N' in df_out.columns:
            fz_aero = df_out['Fz_Downforce_N'].values
        else:
            fz_aero = np.zeros(len(df_out))

        mu_utilized = self.compute_grip_utilization(ax, ay, fz_aero)

        df_out['Mu_Utilized'] = mu_utilized

        logging.info("Tire friction model processed successfully.")
        return df_out