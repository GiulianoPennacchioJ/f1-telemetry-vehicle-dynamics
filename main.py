import os
import logging
from core import F1DataLoader, SpatialResampler, SignalProcessor
from physics import AeroModel, TireModel, GGDiagramAnalyzer, ERSAnalyzer, DRSAnalyzer
from reports import TelemetryComparator, LapReportGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def run_driver_pipeline(session, driver_code: str, config_path: str):
    """Esegue la catena cinematica e fisica completa per un singolo pilota."""
    logging.info(f"--- Processing Telemetry: {driver_code} ---")
    
    loader = F1DataLoader()
    lap = loader.get_driver_fastest_lap(session, driver_code)
    raw_df = loader.extract_raw_telemetry(lap)

    # 1. Spatial Resampling (Δs = 1.0 m)
    resampler = SpatialResampler(step_size_meters=1.0)
    spatial_df = resampler.resample_lap(raw_df)

    # 2. Kinematics & Savitzky-Golay Filtering
    processor = SignalProcessor(window_length=15, poly_order=3)
    kinematic_df = processor.compute_kinematics(spatial_df)

    # 3. Aerodynamics Model
    aero = AeroModel(config_path=config_path)
    aero_df = aero.process_telemetry_aero(kinematic_df)

    # 4. Tire Model & Friction
    tire = TireModel(config_path=config_path)
    tire_df = tire.process_telemetry_tires(aero_df)

    # 5. GG Diagram & G-Forces
    gg = GGDiagramAnalyzer()
    gg_df, _ = gg.process_telemetry_gg(tire_df)

    # 6. ERS Power & Clipping
    ers = ERSAnalyzer(config_path=config_path)
    ers_df = ers.process_telemetry_ers(gg_df)

    # 7. DRS Status
    drs = DRSAnalyzer(config_path=config_path)
    final_df, _ = drs.process_telemetry_drs(ers_df)

    return final_df


def main():
    # Parametri della Sessione
    YEAR = 2025
    LOCATION = 'Monza'
    SESSION_TYPE = 'Q'
    REF_DRIVER = 'NOR'
    COMP_DRIVER = 'VER'
    CONFIG_PATH = "config/car_parameters.json"
    OUTPUT_DIR = "reports/output"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Caricamento Sessione FastF1
    loader = F1DataLoader()
    session = loader.load_session(YEAR, LOCATION, SESSION_TYPE)

    # Step 2: Esecuzione Pipeline Piloti
    df_ref = run_driver_pipeline(session, REF_DRIVER, CONFIG_PATH)
    df_comp = run_driver_pipeline(session, COMP_DRIVER, CONFIG_PATH)

    # Step 3: Comparazione Telemetria
    comparator = TelemetryComparator(driver_ref_name=REF_DRIVER, driver_comp_name=COMP_DRIVER)
    comp_df = comparator.compare_laps(df_ref, df_comp)

    # Step 4: Generazione Grafici e Report Executive
    report_gen = LapReportGenerator(output_dir=OUTPUT_DIR)
    generated_files = report_gen.generate_all_reports(
        df_ref=df_ref,
        df_comp=df_comp,
        comp_df=comp_df,
        ref_name=REF_DRIVER,
        comp_name=COMP_DRIVER
    )

    print("\n==========================================")
    print(" EXECUTION COMPLETED SUCCESSFULLY")
    print("==========================================")
    for report_name, path in generated_files.items():
        print(f"{report_name}: {path}")


if __name__ == "__main__":
    main()