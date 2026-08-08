import os
import logging
from typing import Optional
import pandas as pd
import fastf1


# Configure logging for professional trace visibility
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class F1DataLoader:
    """
    Handles fetching, caching, and preliminary processing of telemetry 
    and lap data from official F1 sessions using FastF1.
    """

    def __init__(self, cache_dir: str = ".fastf1_cache"):
        """
        Initializes the data loader and enables local disk caching.

        Parameters:
        -----------
        cache_dir : str
            Relative or absolute path to the local cache directory.
        """
        self.cache_dir = cache_dir
        self._setup_cache()

    def _setup_cache(self) -> None:
        """Configures FastF1 cache directory if it does not exist."""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            logging.info(f"Created FastF1 cache directory at: {self.cache_dir}")
        fastf1.Cache.enable_cache(self.cache_dir)
        logging.info("FastF1 cache enabled successfully.")

    def load_session(self, year: int, event: str, session_type: str) -> fastf1.core.Session:
        """
        Loads a specific F1 session and fetches telemetry/lap timing data.

        Parameters:
        -----------
        year : int
            Championship season year (e.g., 2025).
        event : str
            Full Grand Prix name, track location, or alias (e.g., 'Monza', 'Italy', 'Italian Grand Prix').
        session_type : str
            Session identifier ('Q' for Qualifying, 'FP2' for Practice 2, 'R' for Race).

        Returns:
        --------
        session : fastf1.core.Session
            Loaded FastF1 session object containing timing and telemetry data.
        """
        # Alias mapping for common short circuit names
        event_mapping = {
            'monza': 'Italy',
            'spa': 'Belgium',
            'silverstone': 'Great Britain',
            'redbullring': 'Austria'
        }
        clean_event = event_mapping.get(event.lower(), event)

        logging.info(f"Fetching session data: {year} {clean_event} GP - Session: {session_type}")
        
        try:
            session = fastf1.get_session(year, clean_event, session_type)
            session.load(telemetry=True, laps=True, weather=False)
            logging.info(f"Session successfully loaded: {session.event['EventName']} {year}")
            return session
        except Exception as err:
            logging.error(f"Failed to load session for {year} {clean_event}: {err}")
            raise err

    def get_driver_fastest_lap(self, session: fastf1.core.Session, driver_code: str) -> fastf1.core.Lap:
        """
        Extracts the single fastest valid lap for a given driver in a session.

        Parameters:
        -----------
        session : fastf1.core.Session
            Loaded session object.
        driver_code : str
            Three-letter driver code (e.g., 'NOR', 'VER').

        Returns:
        --------
        fastest_lap : fastf1.core.Lap
            FastF1 Lap object corresponding to the driver's fastest personal lap.
        """
        driver_laps = session.laps.pick_driver(driver_code)
        
        if driver_laps.empty:
            raise ValueError(f"No lap data found for driver code '{driver_code}' in this session.")

        # Filter for accurate/valid timed laps
        valid_laps = driver_laps.pick_accurate()
        
        if valid_laps.empty:
            logging.warning(f"No accurate laps found for driver {driver_code}. Falling back to all driver laps.")
            valid_laps = driver_laps

        fastest_lap = valid_laps.pick_fastest()
        lap_time_str = str(fastest_lap['LapTime']).split()[-1]
        logging.info(f"Fastest lap for {driver_code}: {lap_time_str} (Lap {fastest_lap['LapNumber']})")
        
        return fastest_lap

    def get_driver_long_run_stint(
        self, 
        session: fastf1.core.Session, 
        driver_code: str, 
        compound: Optional[str] = None
    ) -> fastf1.core.Laps:
        """
        Extracts consecutive laps forming a long run stint (typically FP2 race simulations)
        for tyre degradation and slip energy analysis.

        Parameters:
        -----------
        session : fastf1.core.Session
            Loaded practice/race session object.
        driver_code : str
            Three-letter driver code (e.g., 'NOR', 'VER').
        compound : str, optional
            Filter by tyre compound (e.g., 'MEDIUM', 'HARD', 'SOFT').

        Returns:
        --------
        stint_laps : fastf1.core.Laps
            FastF1 Laps collection containing the sequence of laps in the stint.
        """
        driver_laps = session.laps.pick_driver(driver_code).pick_accurate()

        if compound:
            driver_laps = driver_laps.pick_tyre(compound)

        # Filter out in-laps and out-laps (pit stop entries/exits)
        stint_laps = driver_laps[driver_laps['PitOutTime'].isna() & driver_laps['PitInTime'].isna()]

        logging.info(f"Retrieved long run stint for {driver_code}: {len(stint_laps)} consecutive laps.")
        return stint_laps

    def extract_raw_telemetry(self, lap: fastf1.core.Lap) -> pd.DataFrame:
        """
        Extracts combined high-frequency car data and spatial GPS coordinates 
        from a single lap object.

        Parameters:
        -----------
        lap : fastf1.core.Lap
            Individual Lap object.

        Returns:
        --------
        raw_telemetry : pd.DataFrame
            DataFrame containing time, velocity, pedal inputs, gears, RPM, DRS, and GPS (X, Y).
        """
        telemetry = lap.get_telemetry()

        selected_channels = [
            'Date', 'Time', 'SessionTime', 'Speed', 'RPM', 
            'Gear', 'Throttle', 'Brake', 'DRS', 'X', 'Y', 'Z'
        ]
        
        available_channels = [ch for ch in selected_channels if ch in telemetry.columns]
        raw_telemetry = telemetry[available_channels].copy()

        return raw_telemetry


# Execution block configured for Monza 2025: Norris vs Verstappen
if __name__ == "__main__":
    loader = F1DataLoader()
    
    target_year = 2025
    target_event = 'Monza'
    driver_1 = 'NOR'
    driver_2 = 'VER'
    
    try:
        q_session = loader.load_session(year=target_year, event=target_event, session_type='Q')
        
        nor_lap = loader.get_driver_fastest_lap(q_session, driver_1)
        ver_lap = loader.get_driver_fastest_lap(q_session, driver_2)
        
        telemetry_nor = loader.extract_raw_telemetry(nor_lap)
        telemetry_ver = loader.extract_raw_telemetry(ver_lap)
        
        print("\n==========================================")
        print(f" TELEMETRY FETCH SUCCESSFUL: {target_event} {target_year}")
        print("==========================================")
        print(f" Driver 1 ({driver_1}): {len(telemetry_nor)} samples extracted.")
        print(f" Driver 2 ({driver_2}): {len(telemetry_ver)} samples extracted.")
        print(f" Available Channels: {list(telemetry_nor.columns)}")
        print("==========================================")
        
    except Exception as e:
        print(f"\nExecution Error: {e}")