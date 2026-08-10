import pytest
import numpy as np
import pandas as pd
from physics.ers_analyzer import ERSAnalyzer


@pytest.fixture
def ers_analyzer(tmp_path):
    # Crea un file config JSON temporaneo per isolare il test
    config_file = tmp_path / "car_parameters.json"
    config_file.write_text("""
    {
        "ers": {
            "mguk_max_power_kw": 120.0,
            "clipping_speed_threshold_kmh": 290.0,
            "clipping_ax_threshold_g": 0.05
        }
    }
    """)
    return ERSAnalyzer(config_path=str(config_file))


def test_mguk_power_capping(ers_analyzer):
    # Input di potenza ruote molto elevata
    p_wheels_kw = np.array([100.0, 400.0, 800.0])
    throttle = np.array([50.0, 100.0, 100.0])

    p_mguk = ers_analyzer.estimate_mguk_power(p_wheels_kw, throttle)

    # L'MGU-K non deve mai superare il limite di 120 kW
    assert np.all(p_mguk <= 120.0)
    assert np.all(p_mguk >= 0.0)


def test_clipping_detection_logic(ers_analyzer):
    # Scenario 1: Alta velocità, full gas, accelerazione quasi nulla -> Clipping = True
    # Scenario 2: Bassa velocità, full gas, accelerazione alta -> Clipping = False
    throttle = np.array([100.0, 100.0])
    speed_kmh = np.array([320.0, 150.0])
    ax_g = np.array([0.01, 1.2])

    clipping = ers_analyzer.detect_clipping_events(throttle, speed_kmh, ax_g)

    assert clipping[0] == True
    assert clipping[1] == False