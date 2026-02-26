import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
from pathlib import Path
from typing import List, Optional, Union

# Configuration
DEFAULT_DATA_DIR = "/media/bigdata/PRIMVS/light_curves/"

class PrimvsCatalog:
    def __init__(self, data_dir=DEFAULT_DATA_DIR):
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Catalog directory not found: {self.data_dir}")

    def _resolve_path(self, source_id) -> Path:
        """Resolve the hierarchical path for a given source ID."""
        source_str = str(int(source_id))
        subdir1 = source_str[:3]
        subdir2 = source_str[3:6]
        return self.data_dir / subdir1 / subdir2 / f"{source_str}.csv"

    def get_lightcurve(self, source_id) -> Optional[pd.DataFrame]:
        """
        Retrieve the lightcurve for a given VIRAC Source ID.

        Args:
            source_id (int or str): The unique source identifier.

        Returns:
            pd.DataFrame or None
        """
        filepath = self._resolve_path(source_id)

        if not filepath.exists():
            print(f"Error: Source ID {source_id} not found at {filepath}")
            return None

        try:
            df = pd.read_csv(filepath)
            df['mjd'] = pd.to_numeric(df['mjd'], errors='coerce')

            # Coalesced mag/err columns for convenience
            df['mag'] = df['ks_mag'].fillna(df['z_mag']).fillna(df['y_mag']).fillna(df['j_mag']).fillna(df['h_mag'])
            df['err'] = df['ks_err'].fillna(df['z_err']).fillna(df['y_err']).fillna(df['j_err']).fillna(df['h_err'])

            return df
        except Exception as e:
            print(f"Error reading lightcurve for {source_id}: {e}")
            return None

    def get_lightcurves(self, source_ids) -> dict:
        """
        Retrieve lightcurves for a list of VIRAC Source IDs.

        Args:
            source_ids: list/array of source IDs

        Returns:
            dict: {source_id: DataFrame} for found sources. Missing IDs are omitted.
        """
        results = {}
        missing = []
        for sid in source_ids:
            df = self.get_lightcurve(sid)
            if df is not None:
                results[int(sid)] = df
            else:
                missing.append(sid)

        if missing:
            print(f"{len(missing)} / {len(source_ids)} source IDs not found.")

        return results

    def get_lightcurves_from_fits(self, fits_path, id_column="sourceid", hdu=1) -> dict:
        """
        Extract lightcurves for all VIRAC IDs in a FITS table.

        Args:
            fits_path: Path to the FITS file.
            id_column: Column name containing VIRAC source IDs.
            hdu: HDU index to read from (default 1 for first table extension).

        Returns:
            dict: {source_id: DataFrame}
        """
        from astropy.io import fits as afits
        from astropy.table import Table

        tbl = Table.read(fits_path, hdu=hdu)

        if id_column not in tbl.colnames:
            raise KeyError(
                f"Column '{id_column}' not found in FITS table. "
                f"Available columns: {tbl.colnames}"
            )

        source_ids = tbl[id_column].data
        print(f"Found {len(source_ids)} source IDs in {fits_path}")
        return self.get_lightcurves(source_ids)

    def source_exists(self, source_id) -> bool:
        """Check if a source ID exists in the catalog."""
        return self._resolve_path(source_id).exists()

    def plot_lightcurve(self, source_id, save_path=None):
        """Visualize the lightcurve for a source."""
        df = self.get_lightcurve(source_id)
        if df is None:
            return

        plt.figure(figsize=(10, 6))
        #sns.set_style("whitegrid")

        colors = {'Ks': 'r', 'Z': 'b', 'Y': 'g', 'J': 'orange', 'H': 'brown'}

        for filt in df['filter'].unique():
            subset = df[df['filter'] == filt].dropna(subset=['mag'])
            if len(subset) > 0:
                plt.errorbar(
                    subset['mjd'], subset['mag'], yerr=subset['err'],
                    fmt='o', label=filt, color=colors.get(filt, 'gray'),
                    markersize=4, alpha=0.7
                )

        plt.gca().invert_yaxis()
        plt.title(f"VIRAC Lightcurve: {source_id}")
        plt.xlabel("MJD (Modified Julian Date)")
        plt.ylabel("Magnitude")
        plt.legend()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()


# ==========================================
# CLI
# ==========================================
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python primvs_api.py <source_id>              # single source")
        print("  python primvs_api.py --fits <file.fits>        # batch from FITS")
        sys.exit(1)

    if sys.argv[1] == "--fits":
        fits_path = sys.argv[2]
        id_col = sys.argv[3] if len(sys.argv) > 3 else "sourceid"
        catalog = PrimvsCatalog()
        results = catalog.get_lightcurves_from_fits(fits_path, id_column=id_col)
        print(f"\nRetrieved {len(results)} lightcurves.")
    else:
        source_id = sys.argv[1]
        catalog = PrimvsCatalog()

        print(f"Fetching data for {source_id}...")
        lc = catalog.get_lightcurve(source_id)

        if lc is not None:
            print("\n--- First 5 rows ---")
            print(lc[['mjd', 'filter', 'mag', 'err']].head())
            print(f"\nTotal detections: {len(lc)}")
            print(f"Filters found: {lc['filter'].unique()}")
            try:
                catalog.plot_lightcurve(source_id, save_path=f"lc_{source_id}.png")
            except Exception as e:
                print(f"Could not plot: {e}")
