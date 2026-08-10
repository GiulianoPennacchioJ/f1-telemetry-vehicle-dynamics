import pytest
import numpy as np
import pandas as pd
from core.signal_processing import SignalProcessor


def test_kinematics_curvature_sign():
    """Verifica che la curvatura mantenga il segno (sinistra/destra)."""
    # Creiamo un arco di cerchio (curva a sinistra)
    s = np.linspace(0, 100, 100)
    R = 50.0  # Raggio di curvatura
    theta = s / R
    x = R * np.cos(theta)
    y = R * np.sin(theta)

    df = pd.DataFrame({
        'Distance': s,
        'Speed': np.full_like(s, 100.0),  # Costante 100 km/h
        'X': x,
        'Y': y
    })

    processor = SignalProcessor(window_length=15, poly_order=3)
    kinematic_df = processor.compute_kinematics(df)

    # Verifica presenza delle colonne calcolate
    assert 'a_x_g' in kinematic_df.columns
    assert 'a_y_g' in kinematic_df.columns
    assert 'Curvature' in kinematic_df.columns

    # Per una traiettoria circolare, la curvatura non deve essere piatta a 0 né priva di segno
    assert not np.all(kinematic_df['a_y_g'] == 0)