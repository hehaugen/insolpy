import rpy2.robjects as robjects
from rpy2.robjects.conversion import localconverter
from pathlib import Path
import rpy2.rinterface as ri
import numpy as np
import os
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
            # This returns I think the right value? Does not seem to work after 270 deg...then returns a negative number, appears to be with arctan2 function
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
    return np.minimum.outer(d, d)

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
    sunvector = Vector(np.array(sunvector).flatten())
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
    timeh = 12
    deltat = 1  # hours
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


#with localconverter(config.np_cv_rules):
#    rv = robjects.conversion.py2ri(sv)

#ri.FloatSexpVector((1.2, 3.3, 1.4))
