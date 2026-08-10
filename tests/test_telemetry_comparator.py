import pytest
import numpy as np
import pandas as pd
from reports.telemetry_comparator import TelemetryComparator


@pytest.fixture
def dummy_lap_df():
    s = np.linspace(0, 1000, 100)
    return pd.DataFrame({
        'Distance': s,
        'Speed': 200 + 50 * np.sin(s / 100),
        'Throttle': np.full_like(s, 100.0),
        'Brake': np.zeros_like(s),
        'X': s * 0.5,
        'Y': s * 0.2,
        'Fz_Total_N': np.full_like(s, 15000.0),
        'Mu_Utilized': np.full_like(s, 0.8),
        'Power_Wheels_kW': np.full_like(s, 500.0)
    })


def test_self_comparison_delta_zero(dummy_lap_df):
    comparator = TelemetryComparator(driver_ref_name="NOR", driver_comp_name="NOR_REF")
    comp_df = comparator.compare_laps(dummy_lap_df, dummy_lap_df)

    # Confrontando un giro con se stesso, il delta speed e delta time devono essere zero
    np.testing.assert_allclose(comp_df['Delta_Speed_kmh'], 0.0, atol=1e-5)
    np.testing.assert_allclose(comp_df['Delta_Time_sec'], 0.0, atol=1e-5)


def test_gps_coordinates_preservation(dummy_lap_df):
    comparator = TelemetryComparator(driver_ref_name="NOR", driver_comp_name="VER")
    comp_df = comparator.compare_laps(dummy_lap_df, dummy_lap_df)

    # Verifica presenza delle coordinate GPS per la Track Dominance Map
    assert 'X' in comp_df.columns
    assert 'Y' in comp_df.columns
    assert not comp_df['X'].isnull().any()