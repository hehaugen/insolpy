from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import Union, Optional

import geopandas as gpd
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
from numba import jit
import pandas as pd
import rasterio as rio
from rasterio.warp import transform_bounds, transform
from rasterio.crs import CRS
import numpy as np
import insolation.insolf as insol
from affine import Affine
import xarray as xr
from shapely.geometry import Point, Polygon

# TODO: add documentation

class Vector:
    def __init__(self, vector: np.ndarray):
        if vector.size not in [2, 3]:
            raise ValueError("Array of length 2 or 3 expected, got length of {0}".format(len(vector)))
        self.dim = vector.size
        self.v = vector
        self.mag = np.sqrt(vector.dot(vector))
        self.x = vector[0]
        self.y = vector[1]
        if self.dim == 3:
            self.z = vector[2]
            self.alpha = np.arccos(self.x / self.mag)
            self.beta = np.arccos(self.y / self.mag)
            self.zenith = np.arccos(self.z / self.mag)
            # This returns I think the right value? Does not seem to work after 270 deg...then returns
            # a negative number, appears to be a problem with arctan2 function?
            self.azimuth = (np.pi / 2) + np.arctan2(self.y, self.x)
        else:
            self.azimuth = (np.pi / 2) + np.arctan2(self.y, self.x)

    def is_unit(self):
        if self.mag == 1:
            return True
        else:
            return False


class Dem:
    def __init__(self, array: np.ndarray,
                 transform: Affine,
                 resolution: tuple,
                 crs: rio.crs.CRS,
                 bounds: tuple | rio.coords.BoundingBox,
                 nodata: int | float | None = None,
                 filepath: str | Path | None = None
                 ):
        if nodata is None:
            self.data = array
        else:
            self.data = np.where(array == nodata, np.nan, array)
        self.transform = transform
        self.resolution = resolution
        self.crs = crs
        self.bounds = bounds
        self.filepath = filepath

    @staticmethod
    def load_raster(fl: str | Path):
        with rio.open(fl) as src:
            data = src.read(1)
            transform = src.transform
            resolution = src.res
            bnds = src.bounds
            crs = src.crs
            nodat = src.nodata

        return Dem(data, transform, resolution, crs, bnds, nodat, filepath=fl)


class SunPosCorrections:

    def __init__(self,
                 raster: str | Path | Dem,
                 azimuth_res: float,
                 zenith_res: float,
                 corrections_dset: xr.Dataset | None = None,
                 verbose: bool = False):

        if isinstance(raster, (str, Path)):
            r = Dem.load_raster(raster)
            self.datarray = r.data
            self.data_res = r.resolution[0]
            self.transform = r.transform
            self.crs = r.crs
            self.bounds = list(r.bounds)
            if self.crs != CRS.from_epsg(4326):
                self.latlonbounds = transform_bounds(self.crs, CRS.from_epsg(4326), r.bounds[0], r.bounds[1], r.bounds[2], r.bounds[3])
            else:
                self.latlonbounds = r.bounds
            self.elev_ds_path = raster

        else:
            self.datarray = raster.data
            self.data_res = raster.resolution[0]
            self.transform = raster.transform
            self.crs = raster.crs
            self.bounds = list(raster.bounds)
            if raster.crs != CRS.from_epsg(4326):
                self.latlonbounds = transform_bounds(self.crs, CRS.from_epsg(4326), raster.bounds[0], raster.bounds[1], raster.bounds[2],
                                               raster.bounds[3])
            else:
                self.latlonbounds = raster.bounds
            self.elev_ds_path = raster.filepath

        if self.elev_ds_path is None:
            self.elev_ds_path = []
        elif isinstance(self.elev_ds_path, Path):
            self.elev_ds_path = self.elev_ds_path.as_posix()
        else:
            pass

        self.azimuth_resolution = azimuth_res
        self.zenith_resolution = zenith_res
        self.verbose = verbose
        self._ref_sunpos = self._get_all_sunpos()
        self.resampled_azimuths = None
        self.resampled_zeniths = None
        self._zarr_path = None
        self.sunpos_grid = self._make_sunpos_grid()
        if corrections_dset is None:
            self._terr_cor = None
        else:
            self.corrections = corrections_dset

    @property
    def corrections(self):
        if self._terr_cor is None:
            print("No corrections have been computed yet.")
        else:
            return self._terr_cor

    @corrections.setter
    def corrections(self, dataset: xr.Dataset):
        self._terr_cor = dataset

    # From Evan's code - get_correction_reference() function...needs to have an option to output to zarr file or for
    #   now, hold in memory (for smaller applications). Also, xarray.Dataset, try having separate dimensions with
    #   indices for azimuth and zenith so you can use xarray's "nearest" search capabilities.
    def calculate_terrain_corrections(self, output: str | None = None, outpth: str | Path | None = None):
        """
        Function to calculate the terrain corrections for each sun-position in the resampled sun-position grid using
        the input DEM raster. These calculations can be stored in memory or archived in a zarr database for larger
        raster datasets, using the output argument.

        Args:
            output: str | None
                Currently, the only valid argument is 'zarr' which will append each sun-positions terrain correction
                factors to a zarr file. The default is None, which will store all calculations in-memory. For large
                raster datasets this will likely cause a memory issue.
            outpth: str | Path | None
                Only applicable if output is set to 'zarr', this is the file path location of the output zarr file.
                Must contain the file name with or without .zarr extension.

        Returns: xr.Dataset | None
            Depending on the output option, will return an xarray Dataset or None, as the correction surface for
            each sun-position will be appended to an archive when set to None.

        """
        if output is None:
            cf_array = np.empty((self.datarray.shape[0], self.datarray.shape[1], self.resampled_azimuths.size,
                                 self.resampled_zeniths.size))
            # TODO: remove redundant sun positions
            for i, av in enumerate(self.resampled_azimuths):
                for j, zv in enumerate(self.resampled_zeniths):
                    sv = insol.normalvector(zv, av)
                    shd = fast_doshade(self.datarray, np.array([av, zv]), self.data_res)
                    hs = insol.hillshading(dem=self.datarray, dlxy=self.data_res, sunv=sv)
                    cf = shd * hs
                    cf_array[:, :, i, j] = cf

            corr_xds = xr.Dataset(
                {
                    "correction_factor": (['row', 'column', 'azimuth', 'zenith'], cf_array,
                                          {'description': "Solar Radiation multiplicative correction factor"
                                                          " that accounts for terrain slope and shading.",
                                           'units': "None"})
                },
                coords={
                    "row": np.arange(self.datarray.shape[0]),
                    "column": np.arange(self.datarray.shape[1]),
                    "azimuth": self.resampled_azimuths,
                    "zenith": self.resampled_zeniths
                },
                attrs={
                    'crs': self.crs.to_wkt(),
                    'transform': list(self.transform)[0:6],
                    'bounds': self.bounds,
                    'resolution': [list(self.transform)[0], list(self.transform)[4]],
                    'azimuth_res': self.azimuth_resolution,
                    'zenith_res': self.zenith_resolution,
                    'elev_path': self.elev_ds_path
                }
            )

            self.corrections = corr_xds
            return corr_xds
        elif output == 'zarr':
            if outpth is None:
                raise ValueError("A filepath must be provided when output mode is set to 'zarr'.")
            if isinstance(outpth, str):
                outpth = Path(outpth)
            self._zarr_path = outpth
            template_ds = xr.Dataset(
                {
                    "correction_factor": (['row', 'column', 'azimuth', 'zenith'], np.empty((self.datarray.shape[0], self.datarray.shape[1], self.resampled_azimuths.size,
                                 self.resampled_zeniths.size)),
                                          {'description': "Solar Radiation multiplicative correction factor"
                                                          " that accounts for terrain slope and shading.",
                                           'units': "None"})
                },
                coords={
                    "row": np.arange(self.datarray.shape[0]),
                    "column": np.arange(self.datarray.shape[1]),
                    "azimuth": self.resampled_azimuths,
                    "zenith": self.resampled_zeniths
                },
                attrs={
                    'crs': self.crs.to_wkt(),
                    'transform': list(self.transform)[0:6],
                    'bounds': self.bounds,
                    'resolution': [list(self.transform)[0], list(self.transform)[4]],
                    'azimuth_res': self.azimuth_resolution,
                    'zenith_res': self.zenith_resolution,
                    'elev_path': self.elev_ds_path
                }
            )
            #template_ds = template_ds.chunk({'row': 1000, 'column': 1000})
            template_ds.to_zarr(self._zarr_path.as_posix(), mode='w', compute=False)
            # TODO: remove redundant sun positions
            for i, av in enumerate(self.resampled_azimuths):
                for j, zv in enumerate(self.resampled_zeniths):
                    sv = insol.normalvector(zv, av)
                    shd = fast_doshade(self.datarray, np.array([av, zv]), self.data_res)
                    hs = insol.hillshading(dem=self.datarray, dlxy=self.data_res, sunv=sv)
                    cf = shd * hs
                    cf_ds = xr.Dataset(
                        {
                            "correction_factor": (['row', 'column', 'azimuth', 'zenith'], cf[:,:,None,None])
                        }
                    )
                    #cf_ds = cf_ds.chunk({'row': 1000, 'column': 1000})
                    cf_ds.to_zarr(self._zarr_path.as_posix(),
                                region={'azimuth': slice(i, i + 1), 'zenith': slice(j, j + 1)})

            self.corrections = xr.open_zarr(self._zarr_path)
        else:
            raise NotImplementedError("Output types other than 'zarr' are not supported.")

    def get_terrain_correction(self, azimuth: float, zenith: float) -> xr.DataArray:
        subset = self.corrections.sel(azimuth=azimuth, zenith=zenith, method='nearest').correction_factor

        return subset

    def get_nearest_sunpos(self,
                           azimuth: float | list[float] | np.ndarray,
                           zenith: float | list[float] | np.ndarray,
                           return_ids: bool = False
                           ) -> pd.DataFrame:
        """Returns a dataframe of the interpolated, or nearest, sun-position.

        Args:
            azimuth:
            zenith:
            return_ids:

        Returns:

        """
        aidx = np.searchsorted(self.resampled_azimuths, azimuth)
        aidx = np.where(aidx == self.resampled_azimuths.size, self.resampled_azimuths.size - 1, aidx)
        zidx = np.searchsorted(self.resampled_zeniths, zenith)
        zidx = np.where(zidx == self.resampled_zeniths.size, self.resampled_zeniths.size - 1, zidx)
        df = pd.DataFrame({'azimuth_nearest': self.resampled_azimuths[aidx],
                           'zenith_nearest': self.resampled_zeniths[zidx]})
        if return_ids:
            df['azimuth_idx'] = aidx
            df['zenith_idx'] = zidx

        return df

    def plot_sunpositions(self):
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
        ax1 = axes[0]
        ax2 = axes[1]
        self._ref_sunpos.plot(kind='scatter', x='azimuth', y='zenith',
                              title=f'all possible sun positions (n={len(self._ref_sunpos)})', ax=ax1, s=1)
        self.sunpos_grid.plot(kind='scatter', x='azimuth', y='zenith',
                              title=f'resampled sun positions (n={len(self.sunpos_grid)})', ax=ax2, s=1)
        ax1.yaxis.set_inverted(True)
        ax2.yaxis.set_inverted(True)
        plt.tight_layout()
        plt.show()

    def plot_corrections(self, azimuth: float, zenith: float):
        sel = self.get_terrain_correction(azimuth=azimuth, zenith=zenith)
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
        im = ax.imshow(sel.values, extent=(self.bounds[0], self.bounds[2], self.bounds[1], self.bounds[3]), cmap='gray')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_title(f"azimuth={sel['azimuth'].values}, zenith={sel['zenith'].values}")
        plt.colorbar(im)
        plt.show()

    def _get_all_sunpos(self):
        min_bound = sunpos_timeseries(self.latlonbounds[1], self.latlonbounds[0], '2024-01-01 00:00:00', '2024-12-31 23:59:00', freq='10min')
        max_bound = sunpos_timeseries(self.latlonbounds[3], self.latlonbounds[2], '2024-01-01 00:00:00', '2024-12-31 23:59:00', freq='10min')
        loc_sunpos = pd.concat([min_bound, max_bound]).reset_index(drop=True)

        return loc_sunpos

    def _make_sunpos_grid(self):
        """

        Args:

        Returns:

        """
        min_z = self._ref_sunpos.loc[self._ref_sunpos['zenith'].idxmin(), 'zenith']
        max_z = self._ref_sunpos.loc[self._ref_sunpos['zenith'].idxmax(), 'zenith']
        min_a = self._ref_sunpos.loc[self._ref_sunpos['azimuth'].idxmin(), 'azimuth']
        max_a = self._ref_sunpos.loc[self._ref_sunpos['azimuth'].idxmax(), 'azimuth']

        z_rng = np.linspace(min_z, max_z, int((max_z - min_z) / self.zenith_resolution), True)
        a_rng = np.linspace(min_a, max_a, int((max_a - min_a) / self.azimuth_resolution), True)
        az, zn = np.meshgrid(a_rng, z_rng)

        allsps = gpd.points_from_xy(self._ref_sunpos['azimuth'], self._ref_sunpos['zenith'])
        spgrd_pnts = gpd.points_from_xy(az.ravel(), zn.ravel())
        sppnt_gdf = gpd.GeoDataFrame(geometry=allsps)
        resamp_sp = gpd.GeoDataFrame(geometry=spgrd_pnts)
        concv_rng = np.linspace(sppnt_gdf.geometry.x.min(), sppnt_gdf.geometry.x.max(),
                                int(sppnt_gdf.geometry.x.max() - sppnt_gdf.geometry.x.min()), True)
        points = []
        for v in np.arange(concv_rng.size - 1):
            samp = sppnt_gdf.loc[(sppnt_gdf.geometry.x > concv_rng[v]) & (sppnt_gdf.geometry.x < concv_rng[v + 1])]
            min_pnt = samp.loc[samp.geometry.y.idxmin()]
            max_pnt = samp.loc[samp.geometry.y.idxmax()]
            points.append(min_pnt.geometry)
            points.append(max_pnt.geometry)

        outer_poly_pnts = gpd.GeoDataFrame(geometry=points)
        outer_poly_pnts['normalized_x'] = outer_poly_pnts.geometry.x - (outer_poly_pnts.geometry.x.mean())
        outer_poly_pnts['normalized_y'] = outer_poly_pnts.geometry.y - (outer_poly_pnts.geometry.y.mean() - 15)
        normx = outer_poly_pnts['normalized_x'].values
        normy = outer_poly_pnts['normalized_y'].values
        outer_poly_pnts['sort_angle'] = np.arctan2(normy, normx)
        outer_poly_pnts = outer_poly_pnts.sort_values(by='sort_angle')
        msk = Polygon([[p.x, p.y] for p in outer_poly_pnts.geometry.values])
        clip_sps = resamp_sp.loc[resamp_sp.intersects(msk), :]

        if self.verbose:
            print(f'{len(clip_sps)} of {len(sppnt_gdf)} points retained')

        fdf = pd.DataFrame({'azimuth': clip_sps.geometry.x, 'zenith': clip_sps.geometry.y})
        self.resampled_azimuths = np.round(a_rng[np.isin(a_rng, fdf['azimuth'].unique())], 2)
        self.resampled_zeniths = np.round(z_rng[np.isin(z_rng, fdf['zenith'].unique())], 2)

        return fdf.round(2)

    @staticmethod
    def load_from_dataset(dset: str | Path | xr.Dataset):
        if isinstance(dset, (str, Path)):
            try:
                dset = xr.open_dataset(dset)
            except PermissionError:
                dset = xr.open_zarr(dset)

        tfrm = Affine(*dset.attrs['transform'])
        bounds = tuple(dset.attrs['bounds'])
        resolution = tuple(dset.attrs['resolution'])
        crs = CRS.from_wkt(dset.attrs['crs'])
        elev_fp = dset.attrs['elev_path']
        az_res = dset.attrs['azimuth_res']
        zn_res = dset.attrs['zenith_res']
        if len(dset.attrs['elev_path']) == 0:
            array = dset.isel(azimuth=0, zenith=0).correction_factor.values
            dem_arr = Dem(array, tfrm, resolution, crs, bounds)
        else:
            dem_arr = Dem.load_raster(Path(dset.attrs['elev_path']))

        spcor = SunPosCorrections(dem_arr, az_res, zn_res, corrections_dset=dset)

        return spcor


def sunpos_timeseries(lat: float,
                      lon: float,
                      start: str,
                      end: str,
                      freq: str = '1min',
                      timezone: int = -7,
                      return_julian: bool = False):
    """

    Args:
        lat:
        lon:
        start:
        end:
        freq:
        timezone:
        return_julian:

    Returns:

    """
    dates = pd.date_range(start, end, freq=freq)
    jd = insol.julian_day(dates.year.values, dates.month.values, dates.day.values, dates.hour.values,
                          dates.minute.values)
    dayl = insol.daylength(lat, lon, jd, -7)
    sunrise = dayl[0, :]
    sunset = dayl[1, :]
    shour = dates.hour.values + (dates.minute.values / 60.0) + (dates.second.values / 3600.0)
    day_hr_idx = np.where((sunrise < shour) & (sunset > shour))
    sv = insol.sunvector(jd, lat, lon, -7)
    sp = insol.sunpos(sv)
    azi = sp[0]
    zen = sp[1]
    df = pd.DataFrame({'date': dates, 'zenith': zen, 'azimuth': azi})
    if return_julian:
        df['julian_day'] = jd
    df = df.loc[day_hr_idx[0], :]
    df = df.reset_index().drop(columns='index')

    return df

def inv_sun_angles(sp):
    inv_az = (sp[0] + 180) % 360
    inv_zen = 90 - sp[1]
    return np.array([inv_az, inv_zen])

def pyramid(n):
    r = np.arange(n)
    d = np.minimum(r, r[::-1])
    return np.outer(d, d)

# This was adapted from the FORTRAN function included with the R - insoply package.
# Keep for now
def doshade(dem, sv, res):
    """
    Function to compute terrain shading for a given sun position.
    :param dem: np.ndarray - an elevation array from a DEM
    :param sv: np.array - an array of length 3 representing x, y, z unit vector coordinates of the sun
    :param res: float - DEM resolution
    :return: np.array - Essentially a boolean array of no shade (1) and shade (0)
    """

    sunvec = sv  # from sunvector R function
    dl = res
    sunvec = Vector(np.array(sunvec).flatten()) # could probably do this under the insolpy.sunvector() func
    sombra = np.ones(dem.shape)
    vectororigin = np.zeros(3)
    nrows = dem.shape[0]
    ncols = dem.shape[1]

    inversesunvector = Vector(-sunvec.v / np.max(np.abs([sunvec.x, sunvec.y])))

    sp = insol.sunpos(sv)
    inv_sp = inv_sun_angles(sp)
    normalsunvector = Vector(np.array(insol.normalvector(inv_sp[1], inv_sp[0])).flatten())

    casx = int(1e6 * sunvec.x)
    casy = int(1e6 * sunvec.y)
    if casx < 0:
        f_i = 0
    else:
        f_i = ncols - 1

    if casy < 0:
        f_j = 0
    else:
        f_j = nrows - 1

    j = f_j
    # print("Doing columns")
    for i in range(ncols):
        n = 0
        zcompare = -1e13
        while n >= 0:
            dx = inversesunvector.x * n
            dy = inversesunvector.y * n
            idx = int(i + dx)
            jdy = int(j + dy)
            if ((idx < 0) | (idx > ncols - 1)) | ((jdy < 0) | (jdy > nrows - 1)):
                # print("i While break at n={0}".format(n))
                break
            vectororigin[0] = dx * dl
            vectororigin[1] = dy * dl
            vectororigin[2] = dem[jdy, idx]
            zprojection = np.dot(vectororigin, normalsunvector.v)
            # x_int[jdy, idx] = vectororigin[0]
            # y_int[jdy, idx] = vectororigin[1]
            # zint[jdy, idx] = vectororigin[2]
            # zproj[jdy, idx] = zprojection
            if np.isnan(zprojection):
                sombra[jdy, idx] = np.nan
            elif zprojection < zcompare:
                sombra[jdy, idx] = 0
            else:
                zcompare = zprojection

            n = n + 1

    i = f_i
    # print("Doing rows")
    for j in range(nrows):
        n = 0
        zcompare = -1e13
        while n >= 0:
            dx = inversesunvector.x * n
            dy = inversesunvector.y * n
            idx = int(i + dx)
            jdy = int(j + dy)
            if ((idx < 0) | (idx > ncols - 1)) | ((jdy < 0) | (jdy > nrows - 1)):
                # print("j While break at n={0}".format(n))
                break
            vectororigin[0] = dx * dl
            vectororigin[1] = dy * dl
            vectororigin[2] = dem[jdy, idx]
            zprojection = np.dot(vectororigin, normalsunvector.v)
            # x_int[jdy, idx] = vectororigin[0]
            # y_int[jdy, idx] = vectororigin[1]
            # zint[jdy, idx] = vectororigin[2]
            # zproj[jdy, idx] = zprojection
            if np.isnan(zprojection):
                sombra[jdy, idx] = np.nan
            elif zprojection < zcompare:
                sombra[jdy, idx] = 0
            else:
                zcompare = zprojection

            n = n + 1

    return sombra


@jit(nopython=True)
def fast_doshade(dem: np.ndarray, sp: np.ndarray, res: float):

    dem = dem.astype(np.float64)
    rads = np.radians(sp)
    nvx = np.sin(rads[0]) * np.sin(rads[1])
    nvy = -np.cos(rads[0]) * np.sin(rads[1])
    nvz = np.cos(rads[1])
    sunvec = np.array([nvx, nvy, nvz])
    dl = res
    sombra = np.ones(dem.shape)
    vectororigin = np.zeros(3)
    nrows = dem.shape[0]
    ncols = dem.shape[1]

    inversesunvector = -sunvec / np.max(np.abs(np.array([sunvec[0], sunvec[1]])))

    inv_az = (sp[0] + 180) % 360
    inv_zen = 90 - sp[1]
    inv_sp = np.array([inv_az, inv_zen])
    radsinv = np.radians(inv_sp)
    invvx = np.sin(radsinv[0]) * np.sin(radsinv[1])
    invvy = -np.cos(radsinv[0]) * np.sin(radsinv[1])
    invvz = np.cos(radsinv[1])
    normalsunvector = np.array([invvx, invvy, invvz])

    casx = np.trunc(1e6 * sunvec[0])
    casy = np.trunc(1e6 * sunvec[1])
    if casx < 0:
        f_i = 0
    else:
        f_i = ncols - 1

    if casy < 0:
        f_j = 0
    else:
        f_j = nrows - 1

    j = f_j
    for i in range(ncols):
        n = 0
        zcompare = -1e13
        while n >= 0:
            dx = inversesunvector[0] * n
            dy = inversesunvector[1] * n
            idx = int(i + dx)
            jdy = int(j + dy)
            if ((idx < 0) | (idx > ncols - 1)) | ((jdy < 0) | (jdy > nrows - 1)):
                break
            vectororigin[0] = dx * dl
            vectororigin[1] = dy * dl
            vectororigin[2] = dem[jdy, idx]
            zprojection = np.dot(vectororigin, normalsunvector)
            if np.isnan(zprojection):
                sombra[jdy, idx] = np.nan
            elif zprojection < zcompare:
                sombra[jdy, idx] = 0
            else:
                zcompare = zprojection

            n = n + 1

    i = f_i
    for j in range(nrows):
        n = 0
        zcompare = -1e13
        while n >= 0:
            dx = inversesunvector[0] * n
            dy = inversesunvector[1] * n
            idx = int(i + dx)
            jdy = int(j + dy)
            if ((idx < 0) | (idx > ncols - 1)) | ((jdy < 0) | (jdy > nrows - 1)):
                break
            vectororigin[0] = dx * dl
            vectororigin[1] = dy * dl
            vectororigin[2] = dem[jdy, idx]
            zprojection = np.dot(vectororigin, normalsunvector)
            if np.isnan(zprojection):
                sombra[jdy, idx] = np.nan
            elif zprojection < zcompare:
                sombra[jdy, idx] = 0
            else:
                zcompare = zprojection

            n = n + 1

    return sombra


@jit(nopython=True)
def fast_shade_points(dem: np.ndarray,
                      dem_res: float,
                      irow: np.ndarray,
                      jcol: np.ndarray,
                      sunpos: np.ndarray):
    """ Calculate shaded status (0 or 1) of points at given solar angle(s).

    This function is designed to be called by fast_doshade_points, which handles
    error checking and pre-processing before passing data to this optimized function.

    Args:
        dem: DEM as 2D numpy array of floats
        dem_res: resolution in meters of the DEM
        irow: x-coordinates of locations transformed to integer indices of corresponding DEM cell
        jcol: y-coordinates of locations transformed to integer indices of corresponding DEM cell
        sunpos: 2D Nx2 array where N=number of sun positions in form [zenith, azimuth]

    Returns:
            shdarray: 2D NxM array where N=number of sun positions and M=number of points
    """
    # dem = dem.astype(np.float64)  # to be safe

    shd_locs = np.empty((len(sunpos), len(irow)))
    ang = 0
    for sp in sunpos:
        zn = sp[0]
        az = sp[1]
        znr = np.radians(zn)
        azr = np.radians(az)
        nvx = np.sin(azr) * np.sin(znr)
        nvy = -np.cos(azr) * np.sin(znr)
        nvz = np.cos(znr)
        sv = np.array([nvx, nvy, nvz])

        for i in range(len(irow)):
            xv = sv[0]
            yv = sv[1]
            zv = sv[2]
            nrows = dem.shape[0]
            ncols = dem.shape[1]
            ray_ln = np.max(np.array(dem.shape))
            d = np.sqrt(np.sqrt(1) / (xv ** 2 + yv ** 2))
            dr = np.arange(0, ray_ln, d)
            xr = (dr * xv).astype(np.int32)
            yr = (dr * yv).astype(np.int32)
            zr = dr * dem_res * zv

            zorigin = dem[irow[i], jcol[i]]
            xis = xr + irow[i]
            yjs = yr + jcol[i]
            zproj = zr + zorigin

            lnmsk = np.where(((xis > 0) & (xis < nrows - 1)) & ((yjs > 0) & (yjs < ncols - 1)))[0]
            xi_clp = xis[lnmsk]
            yj_clp = yjs[lnmsk]

            z_ext = []
            for j in range(len(lnmsk)):
                z_ext.append(dem[xi_clp[j], yj_clp[j]])
            z_ext = np.array(z_ext)

            zdiff = zproj[lnmsk] - z_ext

            if (zdiff < 0).any():
                shd = 0.0
            else:
                shd = 1.0

            shd_locs[ang, i] = shd
        ang = ang + 1

    return shd_locs


def fast_doshade_points(raster: str | Path | insolpy.Dem,
                        geom: gpd.GeoDataFrame,
                        sunpos: tuple | np.ndarray = (45.0, 315.0)):
    """ Calculate shaded status (0 or 1) of points at given solar angle(s).

    Can only make calculations off of sun angles, not datetimes. It is designed
    to handle error checking and pre-processing before passing data to the
    'jitted' fast_shade_points function.

    Args:
        raster: DEM
        geom: list of points to determine shading at, in same crs as raster
        sunpos: 2D Nx2 array where N=number of sun positions in form [zenith, azimuth]

    Returns:
        shdarray: 2D NxM array where N=number of sun positions and M=number of points
    """

    if isinstance(raster, (str, Path)):
        r = insolpy.Dem.load_raster(raster)
        datarray = r.data
        data_res = r.resolution[0]
        transform = r.transform
        bounds = r.bounds
    else:
        datarray = raster.data
        data_res = raster.resolution[0]
        transform = raster.transform
        bounds = raster.bounds

    if ((geom.geometry.x < bounds.left).any()) or ((geom.geometry.x > bounds.right).any()) or (
    (geom.geometry.y < bounds.bottom).any()) or ((geom.geometry.y > bounds.top).any()):
        raise ValueError("An input point is not within the bounds of the raster dataset.")

    if (geom.geom_type != 'Point').any():
        raise AttributeError("One of the input geometries is not of geom_type POINT.")

    gx = geom.geometry.x.values
    gy = geom.geometry.y.values

    if len(gx) != len(gy):
        raise ValueError("The coordinate arrays do not match.")

    gx, gy = rio.transform.rowcol(transform, gx, gy)

    if isinstance(sunpos, tuple):
        sunpos = np.array([sunpos])

    shdarray = fast_shade_points(datarray, data_res, gx, gy, sunpos)

    return shdarray


def shade_at_points(dem: np.ndarray,
                    dem_res: float,
                    xcoords: np.ndarray,
                    ycoords: np.ndarray,
                    transform: Affine,
                    sunvec: np.ndarray):
    """

    Args:
        dem:
        dem_res:
        xcoords:
        ycoords:
        transform:
        sunvec:

    Returns:

    """
    if len(xcoords) != len(ycoords):
        raise ValueError("The coordinate arrays do not match.")

    if sunvec.shape[0] != len(xcoords):
        raise ValueError("The number of sunvectors does not match the number of points.")

    irow, jcol = rio.transform.rowcol(transform, xcoords, ycoords)

    shd_locs = []
    for i in range(len(xcoords)):
        xv = sunvec[i,0]
        yv = sunvec[i,1]
        zv = sunvec[i,2]
        nrows, ncols = dem.shape
        ray_ln = np.max(np.array(dem.shape))
        d = np.sqrt(np.sqrt(1) / (xv ** 2 + yv ** 2))
        dr = np.arange(0, ray_ln, d)
        xr = (dr * xv).astype(np.int32)
        yr = (dr * yv).astype(np.int32)
        zr = dr * dem_res * zv

        zorigin = dem[irow[i], jcol[i]]
        xis = xr + irow[i]
        yjs = yr + jcol[i]
        zproj = zr + zorigin
        lnmsk = np.where(((xis > 0) & (xis < nrows - 1)) & ((yjs > 0) & (yjs < ncols - 1)))
        xi_clp = xis[lnmsk[0]]
        yj_clp = yjs[lnmsk[0]]
        z_ext = dem[xi_clp, yj_clp]

        zdiff = zproj[lnmsk[0]] - z_ext

        if (zdiff < 0).any():
            shd = 0.0
        else:
            shd = 1.0

        shd_locs.append(shd)

    return np.array(shd_locs)

def doshade_points(raster: str | Path | Dem,
                   geom: gpd.GeoDataFrame,
                   dttimes: datetime | pd.Timestamp | pd.DatetimeIndex | None = None,
                   sunpos: tuple | np.ndarray = (45.0, 315.0),
                   timezn: int | float = -7):

    if isinstance(raster, (str, Path)):
        r = Dem.load_raster(raster)
        datarray = r.data
        data_res = r.resolution[0]
        transform = r.transform
        bounds = r.bounds
    else:
        datarray = raster.data
        data_res = raster.resolution[0]
        transform = raster.transform
        bounds = raster.bounds

    if ((geom.geometry.x < bounds.left).any()) or ((geom.geometry.x > bounds.right).any()) or ((geom.geometry.y < bounds.bottom).any()) or ((geom.geometry.y > bounds.top).any()):
        raise ValueError("An input point is not within the bounds of the raster dataset.")

    if (geom.geom_type != 'Point').any():
        raise AttributeError("One of the input geometries is not of geom_type POINT.")

    gwgs84 = geom.to_crs(4326)
    glat = gwgs84.geometry.y.values
    glon = gwgs84.geometry.x.values
    gx = geom.geometry.x.values
    gy = geom.geometry.y.values

    if isinstance(dttimes, (datetime, pd.Timestamp)):
        print("Dates have been provided, using dates over sun position argument.")
        jd = insol.julian_day(dttimes.year, dttimes.month, dttimes.day, dttimes.hour, dttimes.minute, dttimes.second)
        dayl = insol.daylength(glat, glon, jd, timezn)
        sunrise = dayl[0, :]
        sunset = dayl[1, :]
        shour = dttimes.hour + (dttimes.minute / 60.0) + (dttimes.second / 3600.0)
        sv = insol.sunvector(jd, glat, glon, timezn).T
        shd = shade_at_points(datarray, data_res, gx, gy, transform, sv)
        shd = np.where((sunrise < shour) & (sunset > shour), shd, 0.0)
        shdarray = np.array([shd])

    elif isinstance(dttimes, pd.DatetimeIndex):
        print("Dates have been provided, using dates over sun position argument.")
        jds = insol.julian_day(dttimes.year.values, dttimes.month.values, dttimes.day.values, dttimes.hour.values, dttimes.minute.values, dttimes.second.values)
        shd_arrs = []
        for n, d in enumerate(jds):
            dayl = insol.daylength(glat, glon, d, timezn)
            sunrise = dayl[0,:]
            sunset = dayl[1,:]
            shour = dttimes.hour.values[n] + (dttimes.minute.values[n] / 60.0) + (dttimes.second.values[n] / 3600.0)
            sv = insol.sunvector(d, glat, glon, timezn).T
            shd = shade_at_points(datarray, data_res, gx, gy, transform, sv)
            shd = np.where((sunrise < shour) & (sunset > shour), shd, 0.0)
            shd_arrs.append(shd)
        shdarray = np.array(shd_arrs)
    else:
        if isinstance(sunpos, tuple):
            sunpos = np.array([sunpos])
            # sunvecs = np.array(insol.normalvector(sunpos[:,0], sunpos[:,1]))

        sunvecs = insol.normalvector(sunpos[:,0], sunpos[:,1]).T
        shd_arrs = []
        for s in sunvecs:
            sv = np.tile(s, (len(gx), 1))
            shd = shade_at_points(datarray, data_res, gx, gy, transform, sv)
            shd_arrs.append(shd)
        shdarray = np.array(shd_arrs)

    return shdarray


def hillshade_points(raster: str | Path | Dem,
                     geom: gpd.GeoDataFrame,
                     dttimes: datetime | pd.Timestamp | pd.DatetimeIndex | None = None,
                     sunpos: tuple | np.ndarray = (45.0, 315.0),
                     timezn: int | float = -7):
    """

    Args:
        raster:
        geom:
        dttimes:
        sunpos:
        timezn:

    Returns:

    """
    if isinstance(raster, (str, Path)):
        r = Dem.load_raster(raster)
        datarray = r.data
        data_res = r.resolution[0]
        transform = r.transform
        bounds = r.bounds
    else:
        datarray = raster.data
        data_res = raster.resolution[0]
        transform = raster.transform
        bounds = raster.bounds

    if ((geom.geometry.x < bounds.left).any()) or ((geom.geometry.x > bounds.right).any()) or ((geom.geometry.y < bounds.bottom).any()) or ((geom.geometry.y > bounds.top).any()):
        raise ValueError("An input point is not within the bounds of the raster dataset.")

    if (geom.geom_type != 'Point').any():
        raise AttributeError("One of the input geometries is not of geom_type POINT.")

    gwgs84 = geom.to_crs(4326)
    glat = gwgs84.geometry.y.values
    glon = gwgs84.geometry.x.values
    gx = geom.geometry.x.values
    gy = geom.geometry.y.values
    irow, jcol = rio.transform.rowcol(transform, gx, gy)

    if isinstance(dttimes, (datetime, pd.Timestamp)):
        print("Dates have been provided, using dates over sun position argument.")
        jd = insol.julian_day(dttimes.year, dttimes.month, dttimes.day, dttimes.hour, dttimes.minute, dttimes.second)
        dayl = insol.daylength(glat, glon, jd, timezn)
        sunrise = dayl[0, :]
        sunset = dayl[1, :]
        shour = dttimes.hour + (dttimes.minute / 60.0) + (dttimes.second / 3600.0)
        sv = insol.sunvector(jd, glat, glon, timezn).T
        hs_arr = []
        for i in range(len(gx)):
            if (sunrise[i] < shour) & (sunset[i] > shour):
                garr = datarray[irow[i] - 1:irow[i] + 2, jcol[i] - 1:jcol[i] + 2]
                hs = insol.hillshading(garr, data_res, sv[i,:])
                hillshd = hs[1, 1]
            else:
                hillshd = 0.0
            hs_arr.append(hillshd)
        hsarray = np.array([hs_arr])
    elif isinstance(dttimes, pd.DatetimeIndex):
        print("Dates have been provided, using dates over sun position argument.")
        jds = insol.julian_day(dttimes.year.values, dttimes.month.values, dttimes.day.values, dttimes.hour.values, dttimes.minute.values, dttimes.second.values)
        hs_arr = []
        for n, d in enumerate(jds):
            dayl = insol.daylength(glat, glon, d, timezn)
            sunrise = dayl[0, :]
            sunset = dayl[1, :]
            shour = dttimes.hour.values[n] + (dttimes.minute.values[n] / 60.0) + (dttimes.second.values[n] / 3600.0)
            sv = insol.sunvector(d, glat, glon, timezn).T
            loc_arr = []
            for i in range(len(gx)):
                if (sunrise[i] < shour) & (sunset[i] > shour):
                    garr = datarray[irow[i] - 1:irow[i] + 2, jcol[i] - 1:jcol[i] + 2]
                    hs = insol.hillshading(garr, data_res, sv[i, :])
                    hillshd = hs[1, 1]
                else:
                    hillshd = 0.0
                loc_arr.append(hillshd)
            hs_arr.append(loc_arr)
        hsarray = np.array(hs_arr)
    else:
        if isinstance(sunpos, tuple):
            sunpos = np.array([sunpos])

        sunvecs = insol.normalvector(sunpos[:,0], sunpos[:,1]).T
        hs_arr = []
        for s in sunvecs:
            sv = np.tile(s, (len(gx), 1))
            loc_arr = []
            for i in range(len(gx)):
                garr = datarray[irow[i] - 1:irow[i] + 2, jcol[i] - 1:jcol[i] + 2]
                hs = insol.hillshading(garr, data_res, sv[i, :])
                hillshd = hs[1, 1]
                loc_arr.append(hillshd)
            hs_arr.append(loc_arr)
        hsarray = np.array(hs_arr)

    return hsarray


def dailyshade(dem_arr, res, lat, lon, timezone, start, end):
    """
    Computes average daily shading factor for every cell of a digital elevation model (DEM) for a range of dates.
    :param dem_arr: np.array - a DEM/raster dataset as an array
    :param res: float - resolution of the input DEM
    :param lat: float - decimal degree of a representative latitude for the DEM
    :param lon: float - decimal degree of a representative longitude for the DEM
    :param timezone: int - UTC offset of the DEM location
    :param start: str - date string formatted as "YYYY-MM-DD"
    :param end: str - date string formatted as "YYYY-MM-DD"
    :return: tuple - of a 3D numpy.array where axis=2 is time, and the associated list of days
    """

    start_d = datetime.strptime("{0}".format(start), "%Y-%m-%d")
    end_d = datetime.strptime("{0}".format(end), "%Y-%m-%d")
    tdiff = (end_d - start_d).days
    dates = [start_d + timedelta(days=x) for x in range(tdiff + 1)]
    tmzn = timezone

    shd_arrays = []
    for day in dates:
        dlday = day + timedelta(hours=12)
        midjd = insol.julian_day(dlday.year, dlday.month, dlday.day, dlday.hour, dlday.minute, dlday.second)
        day_len = insol.daylength(lat, lon, midjd, tmzn)
        day_arrays = []
        for h in np.linspace(day_len[0], day_len[1], int(day_len[2]), endpoint=True):
            hhours = int(h)
            hminutes = h * 60 % 60

            dayhr = day + timedelta(hours=hhours, minutes=hminutes)
            jd_dayhr = insol.julian_day(dayhr.year, dayhr.month, dayhr.day, dayhr.hour, dayhr.minute, dayhr.second)
            sv = insol.sunvector(dayhr, lat, lon, tmzn)
            sunp = insol.sunpos(sv)

            hsh = insol.hillshading(dem_arr, res, sv)
            shd = fast_doshade(dem_arr, insol.sunpos(sv), res=res)
            hs_fin = hsh * shd

            day_arrays.append(hs_fin)

        day_array = np.dstack(day_arrays)

        day_array = day_array.mean(axis=2)
        shd_arrays.append(day_array)

    return shd_arrays, dates
