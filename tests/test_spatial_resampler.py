import pytest
import numpy as np
import pandas as pd
from core.spatial_resampler import SpatialResampler


@pytest.fixture
def dummy_raw_telemetry():
    """Genera una telemetria grezza simulata basata sul tempo."""
    np.random.seed(42)
    n_points = 500
    time_s = np.linspace(0, 80, n_points)
    speed_kmh = 200 + 50 * np.sin(time_s / 5)
    
    # Integrazione approssimativa della distanza
    speed_ms = speed_kmh / 3.6
    dt = np.diff(time_s, prepend=0)
    distance = np.cumsum(speed_ms * dt)

    df = pd.DataFrame({
        'Time': time_s,
        'Distance': distance,
        'Speed': speed_kmh,
        'Throttle': np.random.uniform(0, 100, n_points),
        'Brake': np.random.uniform(0, 100, n_points),
        'X': distance * 0.8,
        'Y': distance * 0.2
    })
    return df


def test_resampling_uniform_step(dummy_raw_telemetry):
    step_size = 1.0
    resampler = SpatialResampler(step_size_meters=step_size)
    resampled_df = resampler.resample_lap(dummy_raw_telemetry)

    # 1. Verifica che la distanza sia equispaziata
    distances = resampled_df['Distance'].values
    delta_s = np.diff(distances)
    
    np.testing.assert_allclose(delta_s, step_size, atol=1e-5)

    # 2. Verifica assenza di NaN
    assert not resampled_df.isnull().values.any()


def test_resampling_monotonicity(dummy_raw_telemetry):
    resampler = SpatialResampler(step_size_meters=1.0)
    resampled_df = resampler.resample_lap(dummy_raw_telemetry)

    # Verifica che la distanza sia strettamente crescente
    assert resampled_df['Distance'].is_monotonic_increasing