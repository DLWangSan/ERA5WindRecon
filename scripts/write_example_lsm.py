"""Generate valid example_data/lsm_era5.nc from example ERA5 grid (replaces Git LFS pointer)."""
import numpy as np
import xarray as xr
import os

ROOT = os.path.join(os.path.dirname(__file__), "..")
era_path = os.path.join(ROOT, "example_data", "ERA5_2026_04.nc")
out_path = os.path.join(ROOT, "example_data", "lsm_era5.nc")

era = xr.open_dataset(era_path, engine="netcdf4")
lat = era["latitude"].values
lon = era["longitude"].values
lsm = np.zeros((len(lat), len(lon)), dtype=np.float32)
ds = xr.Dataset(
    {"lsm": (["latitude", "longitude"], lsm)},
    coords={"latitude": lat, "longitude": lon},
)
ds["lsm"].attrs.update({"long_name": "land_sea_mask", "units": "1"})
ds.to_netcdf(out_path, engine="netcdf4")
print("wrote", out_path, lsm.shape)
