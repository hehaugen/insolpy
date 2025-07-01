from pathlib import Path
import os

from insolation import insolf as insol
import xarray as xr
import pandas as pd
import numpy as np

import insolpy

if __name__ == '__main__':

    spgrid = insolpy.SunPosCorrections.load_from_dataset("D:/GitHub/insolpy/ExampleXX.zarr")
    zn_cf = xr.load_dataset("D:/Modeling/GSFLOW/PRMS_Projects/Upper Yellowstone/prms_grid_solrad_terrain_cf.nc")

    spts = insolpy.sunpos_timeseries(44, -110, "2020-01-01", "2024-12-31", freq="h", return_julian=True)
    dailyts = pd.date_range('2020-01-01', '2024-12-31', freq='D')
    spdtidx = pd.DatetimeIndex(spts['date'])

    daily_cflst = []
    for day in dailyts:
        sel = spts.loc[(spdtidx.year == day.year) & (spdtidx.month == day.month) & (spdtidx.day == day.day), :]
        theor_sr = insol.insolation(sel['zenith'].values, sel['julian_day'].values, 1040, 60, 55, 285.0, 0.02, 0.2)[0]
        grdcf = spgrid.get_nearest_sunpos(sel['azimuth'].values, sel['zenith'].values, return_ids=True)
        polycf = zn_cf.correction_factor.values[:, grdcf['azimuth_idx'].values, grdcf['zenith_idx'].values]
        tot_theor = theor_sr.sum()
        cor_theor = polycf * theor_sr
        fin = cor_theor.sum(axis=1) / tot_theor
        daily_cflst.append(fin)

    dcf = np.array(daily_cflst)

    dcf_ds = xr.Dataset(
        {'correction_factor': (['time', 'FID'], dcf)},
        coords={
            'time': dailyts
        }
    )

    dcf_ds.to_netcdf("D:/Modeling/GSFLOW/PRMS_Projects/Upper Yellowstone/prms_solrad_corrections.nc")