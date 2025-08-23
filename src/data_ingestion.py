# src/data_ingestion.py

import os
import re
import time
import logging
import json
from datetime import datetime

import lightkurve as lk
import numpy as np
import pandas as pd
from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive

# Configure logger for this module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealTransitDataIngestion:
    """
    Handles the download of real exoplanet data from NASA archives
    and TESS light curves from the MAST portal.
    """
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.catalog_dir = os.path.join(self.data_dir, "catalogs")
        os.makedirs(self.catalog_dir, exist_ok=True)

    def load_confirmed_planets_catalog(self):
        """
        Downloads a detailed catalog of all confirmed transiting planets, which
        includes orbital parameters needed for preprocessing.
        """
        try:
            logger.info("Downloading Confirmed Planets Catalog (for parameters)...")
            planets_table = NasaExoplanetArchive.query_criteria(
                table="ps",
                where="tran_flag=1 and tic_id is not null",
                select="pl_name,tic_id,pl_orbper,pl_tranmid,pl_trandur"
            )
            filepath = os.path.join(self.catalog_dir, "confirmed_planets.csv")
            planets_table.to_pandas().to_csv(filepath, index=False)
            logger.info(f"Saved Confirmed Planets Catalog with {len(planets_table)} entries.")
            return planets_table
        except Exception as e:
            logger.error(f"Failed to load Confirmed Planets Catalog: {e}")
            return None

    def load_tess_objects_of_interest(self):
        """
        Downloads a catalog of TESS Objects of Interest (TOIs) that are
        confirmed planets. This is used to select reliable targets for download.
        """
        try:
            logger.info("Downloading TESS Objects of Interest (TOI) catalog (for target selection)...")
            toi_table = NasaExoplanetArchive.query_criteria(
                table="toi",
                where="tfopwg_disp = 'CP' and tid is not null",
                select="toi,tid"
            )
            filepath = os.path.join(self.catalog_dir, "toi_catalog.csv")
            toi_table.to_pandas().to_csv(filepath, index=False)
            logger.info(f"Saved TOI Catalog with {len(toi_table)} entries.")
            return toi_table
        except Exception as e:
            logger.error(f"Failed to load TOI catalog: {e}")
            return None

    def download_tess_lightcurves(self, tic_ids, max_per_star=3):
        """Downloads TESS light curves for a list of TIC IDs."""
        # This function remains the same as your working version
        downloaded_data = []
        raw_dir = os.path.join(self.data_dir, "raw")
        os.makedirs(raw_dir, exist_ok=True)

        for tic_id in tic_ids:
            try:
                target_id_str = f"TIC {tic_id}"
                logger.info(f"Processing target: '{target_id_str}'")
                search_result = lk.search_lightcurve(target_id_str, mission="TESS", author="SPOC")
                if not search_result:
                    search_result = lk.search_lightcurve(target_id_str, mission="TESS")
                if not search_result:
                    logger.error(f"No data found for {target_id_str}")
                    continue

                for i, item in enumerate(search_result):
                    if i >= max_per_star: break
                    lc = item.download()
                    if lc is None: continue
                    save_path = os.path.join(raw_dir, f"TIC_{tic_id}_sector_{lc.sector}.csv")
                    df = pd.DataFrame({
                        'time': lc.time.value, 'flux': lc.flux.value,
                        'flux_err': lc.flux_err.value, 'tic_id': tic_id, 'sector': lc.sector
                    })
                    df.to_csv(save_path, index=False)
                    downloaded_data.append({'tic_id': tic_id, 'sector': lc.sector})
                    logger.info(f"SUCCESS: Downloaded and saved {target_id_str} Sector {lc.sector}")
                    time.sleep(1)
            except Exception as e:
                logger.error(f"Failed to process {target_id_str}: {e}")
        return downloaded_data

def run_data_ingestion(data_dir="data", **kwargs):
    """Orchestrates the data ingestion process."""
    ingestion = RealTransitDataIngestion(data_dir)
    
    # --- DEFINITIVE FIX ---
    # This block ensures BOTH catalogs are downloaded before proceeding.
    logger.info("--- Ensuring All Necessary Catalogs Are Present ---")
    ingestion.load_confirmed_planets_catalog()
    targets = ingestion.load_tess_objects_of_interest()

    if targets is None:
        logger.error("Cannot proceed without a target list from the TOI catalog.")
        return {'real_samples': 0}

    num_stars = kwargs.get('num_stars', 5)
    targets_df = targets.to_pandas()
    tic_ids = targets_df['tid'].dropna().astype(int).unique()
    
    logger.info(f"Selected {num_stars} TIC IDs for download from {len(tic_ids)} available confirmed planets.")
    downloaded = ingestion.download_tess_lightcurves(tic_ids[:num_stars])
    
    return {'real_samples': len(downloaded)}
