import os
import logging
from core import F1DataLoader, SpatialResampler, SignalProcessor
from physics import AeroModel, TireModel, GGDiagramAnalyzer, ERSAnalyzer, DRSAnalyzer
from reports import TelemetryComparator, LapReportGenerator

# Configure logging format and standard log level
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def run_driver_pipeline(session, driver_code: str, config_path: str):
    """
    Executes the full cinematic, dynamic, and physical processing pipeline for a given driver.
    
    Args:
        session: FastF1 Session object.
        driver_code (str): Driver code identifier (e.g., 'NOR', 'VER').
        config_path (str): Path to car vehicle parameters JSON file.

    Returns:
        pd.DataFrame: Fully processed spatial telemetry containing aero, tire, GG, ERS, and DRS channels.
    """
    logging.info(f"--- Processing Telemetry Pipeline for Driver: {driver_code} ---")
    
    # Extract raw lap telemetry
    loader = F1DataLoader()
    lap = loader.get_driver_fastest_lap(session, driver_code)
    raw_df = loader.extract_raw_telemetry(lap)

    # 1. Spatial Resampling (Δs = 1.0 m uniform spatial grid)
    resampler = SpatialResampler(step_size_meters=1.0)
    spatial_df = resampler.resample_lap(raw_df)

    # 2. Kinematics Computation & Savitzky-Golay Filtering
    processor = SignalProcessor(window_length=15, poly_order=3)
    kinematic_df = processor.compute_kinematics(spatial_df)

    # 3. Aerodynamics Force Estimation (Downforce Fz, Aero Drag Fx, L/D ratio)
    aero = AeroModel(config_path=config_path)
    aero_df = aero.process_telemetry_aero(kinematic_df)

    # 4. Tire Model & Friction Coefficient Utilization
    tire = TireModel(config_path=config_path)
    tire_df = tire.process_telemetry_tires(aero_df)

    # 5. GG Diagram Acceleration & Friction Circle Analysis
    gg = GGDiagramAnalyzer()
    gg_df, _ = gg.process_telemetry_gg(tire_df)

    # 6. ERS MGU-K Power & Clipping State Identification
    ers = ERSAnalyzer(config_path=config_path)
    ers_df = ers.process_telemetry_ers(gg_df)

    # 7. DRS Status Evaluation
    drs = DRSAnalyzer(config_path=config_path)
    final_df, _ = drs.process_telemetry_drs(ers_df)

    return final_df


def main():
    # Pipeline execution parameters and directory setup
    YEAR = 2025
    LOCATION = 'Monza'
    SESSION_TYPE = 'Q'
    REF_DRIVER = 'NOR'
    COMP_DRIVER = 'VER'
    CONFIG_PATH = "config/car_parameters.json"
    OUTPUT_DIR = "reports/output"

    # Ensure output destination directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Load F1 Qualifying Session via FastF1
    loader = F1DataLoader()
    session = loader.load_session(YEAR, LOCATION, SESSION_TYPE)

    # Step 2: Execute physics and kinematics pipelines for reference and comparative drivers
    df_ref = run_driver_pipeline(session, REF_DRIVER, CONFIG_PATH)
    df_comp = run_driver_pipeline(session, COMP_DRIVER, CONFIG_PATH)

    # Step 3: Compare telemetry channels and construct differential dataframe (comp_df)
    comparator = TelemetryComparator(driver_ref_name=REF_DRIVER, driver_comp_name=COMP_DRIVER)
    comp_df = comparator.compare_laps(df_ref, df_comp)

    # Step 4: Generate Executive Performance Reports (Direct Method Calls)
    report_gen = LapReportGenerator(output_dir=OUTPUT_DIR)
    
    generated_files = {
        "1. Telemetry Overlay Plot": report_gen.generate_telemetry_overlay(
            comp_df, ref_name=REF_DRIVER, comp_name=COMP_DRIVER
        ),
        "2. Track Dominance Map": report_gen.generate_track_dominance_map(
            comp_df, ref_name=REF_DRIVER, comp_name=COMP_DRIVER
        ),
        "3. GG Diagram Plot": report_gen.generate_gg_diagram_plot(
            df_ref, df_comp, ref_name=REF_DRIVER, comp_name=COMP_DRIVER
        ),
        "4. ERS Clipping Plot": report_gen.generate_ers_clipping_plot(
            df_ref, df_comp, ref_name=REF_DRIVER, comp_name=COMP_DRIVER
        ),
        "5. Aero Forces Plot": report_gen.generate_aero_forces_plot(
            df_ref, df_comp, ref_name=REF_DRIVER, comp_name=COMP_DRIVER
        )
    }

    # Step 5: Output execution status summary
    print("\n==========================================")
    print(" ALL EXECUTIVE REPORTS GENERATED SUCCESSFULLY")
    print("==========================================")
    for report_name, path in generated_files.items():
        print(f"{report_name}: {path}")


if __name__ == "__main__":
    main()