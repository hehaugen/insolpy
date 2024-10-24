import geopandas as gpd
import rioxarray
import pandas as pd
import rasterio as rio
import numpy as np
from datetime import datetime
from datetime import timedelta

import config


class Vector:
    def __init__(self, vector):
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

def sunpos(sv):
    sp = config.R_sunpos.sunpos(sv)
    return np.array(sp).flatten()

def JD(date):
    jd = config.R_JDymd.JDymd(date.year,date.month,date.day,date.hour,date.minute,date.second)
    return jd[0]

def sunvector(jd, lat, lon, timezone):
    sv = config.R_sunvector.sunvector(jd, lat, lon, timezone)
    return sv

def inv_sun_angles(sp):
    inv_az = (sp[0] + 180) % 360
    inv_zen = 90 - sp[1]
    return np.array([inv_az, inv_zen])

def daylength(lat, lon, jd, timezone):
    srss = config.R_daylength.daylength(lat, lon, jd, timezone)
    return np.array(srss).flatten()

def pyramid(n):
    r = np.arange(n)
    d = np.minimum(r, r[::-1])
    return np.outer(d, d)

def normal_vector(slope, aspect):
    nv = config.R_normalvector.normalvector(slope, aspect)
    return Vector(np.array(nv).flatten())

# This was adapted from the FORTRAN function included with the R - insoply package.
def doshade(dem, sv, res):
    """
    Function to compute terrain shading for a given sun position.
    :param dem: np.ndarray - an elevation array from a DEM
    :param sv: np.array - an array of length 3 representing x, y, z unit vector coordinates of the sun
    :param res: float - DEM resolution
    :return: np.array - Essentially a boolean array of no shade (1) and shade (0)
    """
    # TODO - I think this function could have improved performance, if everything was done using vector operations
    #   rather than looping through, tried to figure out a way but haven't solved it yet.
    sunvector = sv  # from sunvector R function
    dl = res
    sunvector = Vector(np.array(sunvector).flatten()) # could probably do this under the insolpy.sunvector() func
    sombra = np.ones(dem.shape)
    vectororigin = np.zeros(3)
    nrows = dem.shape[0]
    ncols = dem.shape[1]

    inversesunvector = Vector(-sunvector.v / np.max(np.abs([sunvector.x, sunvector.y])))

    sp = sunpos(sv)
    inv_sp = inv_sun_angles(sp)
    normalsunvector = Vector(np.array(config.R_normalvector.normalvector(inv_sp[1], inv_sp[0])).flatten())

    casx = int(1e6 * sunvector.x)
    casy = int(1e6 * sunvector.y)
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
            if zprojection < zcompare:
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
            if zprojection < zcompare:
                sombra[jdy, idx] = 0
            else:
                zcompare = zprojection

            n = n + 1

    return sombra


def cgrad(dem, dlx, cArea=False):
    dly = dlx
    grad = np.gradient(dem)
    x_grad = grad[0] * 0.5 * dlx
    y_grad = grad[1] * 0.5 * dly
    z_grad = np.ones(dem.shape)
    z_grad = z_grad * dlx * dly
    cellArea = np.sqrt(x_grad ** 2 + y_grad ** 2 + z_grad ** 2)
    cellgr = np.stack([x_grad, y_grad, z_grad], axis=2)
    if cArea:
        return cellArea
    else:
        for i in np.arange(3):
            cellgr[:, :, i] = cellgr[:, :, i] / cellArea

        return cellgr


def hillshading(cgrad, sv):
    hsh = cgrad[:, :, 0] * sv[0] + cgrad[:, :, 1] * sv[1] + cgrad[:, :, 2] * sv[2]
    hsh = (hsh + np.abs(hsh)) / 2
    return hsh

# TODO - need to include code to check that the geometry is within the provided raster bounds
def doshade_geometry(raster, geom, sun_vector, poly_output='scalar'):
    """
    Application of the insolpy.doshade() function for a point or polygon geometry calculated using 1D vectors from
    the geometry point(s) to see if terrain is shading those point(s).
    :param raster: a rioxarray.DataArray or rioxarray.Dataset of elevation in a projected CRS. Must include a transform
    attribute (DataArray.rio.transform()).
    :param geom: a geopandas.GeoDataFrame of a point or polygon area of interest to calculate shading. This should be a
    GeoDataFrame with only 1 row, if it has more the function will only compute shading for the first entry in the
    dataframe. Must have matching CRS as 'raster' input.
    :param sun_vector: numpy.array of a unit vector representing the sun direction (x,y,z). This is the output from
    the insolpy.sunvector() or insolpy.normalvector() functions.
    :param poly_output: str - 'scalar' for a scalar output which averages all cells in the target polygon, or
    'raster' for a rioxarray.DataArray of shading (the polygon is represented as a raster.
    :return: scalar or rioxarray.DataArray of shading factor (1=not shaded, 0=shaded)
    """
    # raster is a rioxarray DataArray or Dataset of elevation
    # assumes raster and geom are in the same crs
    # geom is a geopandas.GeoDataFrame
    g = geom.geometry.iloc[0]
    datarray = raster.sel(band=1).data
    data_res = raster.rio.resolution()[0]
    transform = raster.rio.transform()

    xv = sun_vector[0]
    yv = sun_vector[1]
    zv = sun_vector[2]
    nrows, ncols = datarray.shape
    ray_ln = np.max(datarray.shape)
    d = np.sqrt(np.sqrt(1) / (xv ** 2 + yv ** 2))
    dr = np.arange(0, ray_ln, d)
    xr = (dr * xv).astype(int)
    yr = (dr * yv).astype(int)
    zr = dr * data_res * zv

    if g.geom_type == 'Point':
        origin = rio.transform.rowcol(transform, g.x, g.y)
        zorigin = datarray[origin[0], origin[1]]
        xis = xr + origin[1]
        yjs = yr + origin[0]
        zproj = zr + zorigin
        lnmsk = np.where(((xis > 0) & (xis < ncols - 1)) & ((yjs > 0) & (yjs < nrows - 1)))
        xi_clp = xis[lnmsk[0]]
        yj_clp = yjs[lnmsk[0]]
        z_ext = datarray[yj_clp, xi_clp]

        zdiff = zproj[lnmsk[0]] - z_ext

        if (zdiff < 0).any():
            shd = 0.0
        else:
            shd = 1.0

    elif g.geom_type == 'Polygon':
        clipped = raster.rio.clip(geom.geometry.values, geom.crs, drop=False)
        origins = np.where(~np.isnan(clipped.sel(band=1).data))
        zorigins = datarray[origins[0], origins[1]]
        xos = origins[1][None, :]
        yos = origins[0][None, :]
        xros = np.tile(xr, (origins[1].shape[0], 1)).T
        xis = (xos + xros)
        yros = np.tile(yr, (origins[1].shape[0], 1)).T
        yjs = (yos + yros)
        xboo = ((xis > 0) & (xis < ncols - 1)).all(axis=1)
        xmsk_sz = xboo[xboo].shape[0]
        xboo = np.tile(xboo, (xis.shape[1], 1)).T
        yboo = ((yjs > 0) & (yjs < nrows - 1)).all(axis=1)
        ymsk_sz = yboo[yboo].shape[0]
        yboo = np.tile(yboo, (yjs.shape[1], 1)).T
        lnmsk = np.logical_and(xboo, yboo)
        msk_rows = np.min([xmsk_sz, ymsk_sz])
        msk_cols = lnmsk.shape[1]
        xi_clp = xis[lnmsk]
        yj_clp = yjs[lnmsk]
        z_ext = datarray[yj_clp, xi_clp]
        z_ext = z_ext.reshape((msk_rows, msk_cols))
        zproj = zorigins[None, :] + np.tile(zr, (zorigins.shape[0], 1)).T
        zdiff = zproj[lnmsk].reshape((msk_rows, msk_cols)) - z_ext
        shdf = np.where((zdiff < 0).any(axis=0), 0.0, 1.0)

        if poly_output == 'scalar':
            shd = np.nanmean(shdf)
        elif poly_output == 'raster':
            shd = clipped.sel(band=1).rename('shade_factor')
            shd.data[origins[0], origins[1]] = shdf
            shd.attrs = {'units': '1=not shaded, 0=shaded'}
            shd = shd[np.min(origins[0]):np.max(origins[0] + 1), np.min(origins[1]):np.max(origins[1] + 1)]
        else:
            print("poly_output argument is not recognized, choose 'scalar' or 'raster'")
            shd = None

    else:
        print("Geometry type given is not compatible.")
        shd = None

    return shd


def hillshade_geometry(raster, geom, sun_vector, poly_output='scalar'):
    """
    Application of the insolpy.hillshading() function for a point or polygon geometry to define the angle adjustment
    for solar radiation based on terrain slope and sun angle.
    :param raster: a rioxarray.DataArray or rioxarray.Dataset of elevation in a projected CRS. Must include a transform
    attribute (DataArray.rio.transform()).
    :param geom: a geopandas.GeoDataFrame of a point or polygon area of interest to calculate shading. This should be a
    GeoDataFrame with only 1 row, if it has more the function will only compute shading for the first entry in the
    dataframe. Must have matching CRS as 'raster' input.
    :param sun_vector: numpy.array of a unit vector representing the sun direction (x,y,z). This is the output from
    the insolpy.sunvector() or insolpy.normalvector() functions.
    :param poly_output: str - 'scalar' for a scalar output which averages all cells in the target polygon, or
    'raster' for a rioxarray.DataArray of shading (the polygon is represented as a raster).
    :return:
    """
    datarray = raster.sel(band=1).data
    g = geom.geometry.iloc[0]
    data_res = raster.rio.resolution()[0]
    transform = raster.rio.transform()

    if g.geom_type == 'Point':
        origin = rio.transform.rowcol(transform, g.x, g.y)
        garr = datarray[origin[0] - 1:origin[0] + 2, origin[1] - 1:origin[1] + 2]
        cgarr = cgrad(garr, data_res)
        hs = hillshading(cgarr, sun_vector)
        hillshd = hs[1, 1]

    elif g.geom_type == 'Polygon':
        clipped = raster.rio.clip(geom.geometry.values, geom.crs)
        clipped = clipped.sel(band=1)
        garr = cgrad(clipped.data, data_res)
        HS = hillshading(garr, sun_vector)

        if poly_output == 'scalar':
            hillshd = np.nanmean(HS)
        elif poly_output == 'raster':
            hillshd = clipped.rename('hillshade_factor')
            hillshd.data = HS
            hillshd.attrs = {'units': 'None'}
        else:
            print("poly_output argument is not recognized, choose 'scalar' or 'raster'")
            hillshd = None

    else:
        print("Geometry type given is not compatible.")
        hillshd = None

    return hillshd


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
        midjd = JD(dlday)
        day_len = daylength(lat, lon, midjd, tmzn)
        day_arrays = []
        for h in np.linspace(day_len[0], day_len[1], int(day_len[2]), endpoint=True):
            hhours = int(h)
            hminutes = h * 60 % 60

            dayhr = JD(day + timedelta(hours=hhours, minutes=hminutes))
            sv = sunvector(dayhr, lat, lon, tmzn)

            cg = cgrad(dem_arr, dlx=res)
            hsh = hillshading(cg, sv)
            shd = doshade(dem_arr, sv, res=res)
            HS = hsh * shd

            day_arrays.append(HS)

        day_array = np.dstack(day_arrays)

        day_array = day_array.mean(axis=2)
        shd_arrays.append(day_array)

    return shd_arrays, dates


def dailyshade_geometry(raster, geom, timezone, start, end):
    """
    Computes average daily shading factor for an input geometry.
    :param raster: a rioxarray.DataArray or rioxarray.Dataset of elevation (1 band) in a projected CRS.
    Must include a transform attribute (DataArray.rio.transform()).
    :param geom: a geopandas.GeoDataFrame of a point or polygon area of interest to calculate shading. This should be a
    GeoDataFrame with only 1 row, if it has more the function will only compute shading for the first entry in the
    dataframe. Must have matching CRS as 'raster' input.
    :param timezone: int - UTC offset of the DEM location
    :param start: str - date string formatted as "YYYY-MM-DD"
    :param end: str - date string formatted as "YYYY-MM-DD"
    :return: pandas.Series - of daily shading factors for the input geometry
    """

    start_d = datetime.strptime("{0}".format(start), "%Y-%m-%d")
    end_d = datetime.strptime("{0}".format(end), "%Y-%m-%d")
    tdiff = (end_d - start_d).days
    dates = [start_d + timedelta(days=x) for x in range(tdiff + 1)]
    tmzn = timezone
    g = geom.geometry.iloc[0]
    res = raster.rio.resolution()[0]

    if g.geom_type == 'Point':
        wgs_geom = geom.to_crs(4326)
        lat = wgs_geom.geometry.y.iloc[0]
        lon = wgs_geom.geometry.x.iloc[0]
    elif g.geom_type == 'Polygon':
        cent = geom.centroid
        cent_wgs = cent.to_crs(4326)
        lat = cent_wgs.geometry.y.iloc[0]
        lon = cent_wgs.geometry.x.iloc[0]
    else:
        print("Geometry type input not supported.")
        lat = None
        lon = None

    shd_vals = []
    for day in dates:
        dlday = day + timedelta(hours=12)
        midjd = JD(dlday)
        day_len = daylength(lat, lon, midjd, tmzn)
        hour_vals = []
        for h in np.linspace(day_len[0], day_len[1], int(day_len[2]), endpoint=True):
            hhours = int(h)
            hminutes = h * 60 % 60

            dayhr = JD(day + timedelta(hours=hhours, minutes=hminutes))
            sv = sunvector(dayhr, lat, lon, tmzn)

            hsh = hillshade_geometry(raster, geom, sv, poly_output='scalar')
            shd = doshade_geometry(raster, geom, sv, poly_output='scalar')
            HS = hsh * shd

            hour_vals.append(HS)

        day_array = np.array(hour_vals)

        day_shd = day_array.mean()
        shd_vals.append(day_shd)

    shd_srs = pd.Series(shd_vals, index=pd.DatetimeIndex(dates))
    shd_srs.name = 'shd_factor'

    return shd_srs

#with localconverter(config.np_cv_rules):
#    rv = robjects.conversion.py2ri(sv)

#ri.FloatSexpVector((1.2, 3.3, 1.4))
