import pynhd as nhd
import py3dep
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta
import numpy as np
import rioxarray

from insolpy import sunvector
from insolpy import JD
from insolpy import sunpos
from insolpy import cgrad
from insolpy import hillshading
from insolpy import doshade
from insolpy import daylength

# Get shading/relief for actual topography
# first get some DEM data for a HUC12 cathment
#huc12 = '100901010101'
#geom = pynhd.WaterData('wbd12').byid('huc12', huc12).geometry[0]
#dem = py3dep.get_dem(geom, 30)
#dem_proj = dem.rio.reproject(5071)
dem = rioxarray.open_rasterio(r'C:\Users\CNB968\OneDrive - MT\Github\insolpy\Examples\100901010101_30m.tif')
huc12 = '100901010101'
dem_bbox = dem.rio.bounds()
lon = (dem_bbox[0] + dem_bbox[2]) / 2
lat = (dem_bbox[1] + dem_bbox[3]) / 2
dem_proj = dem.rio.reproject(5071)

# Find daylight hours for range of dates 2000-01-01 2000-12-31
# loop through daylight hours for each date
# days = []
# for date in dates:
#   get daylight hours for date using R functions
#   day_array = np.array()
#   for hour in daylight hours:
#       calculate shading
#       stack on day_array
#   days.append(day_array.mean(axis=2))
start_d = datetime(2000,1,1,0,0,0)
dates = [start_d + timedelta(days=x) for x in range(366)]
timeh = 12
deltat = 1 # hours
tmzn = -7

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

        cg = cgrad(dem_proj.values[0], dlx=dem_proj.rio.resolution()[0])
        hsh = hillshading(cg, sv)
        shd = doshade(dem_proj.values[0], sv, res=dem_proj.rio.resolution()[0])
        HS = hsh * shd

        day_arrays.append(HS)

    day_array = np.dstack(day_arrays)

    day_array = day_array.mean(axis=2)
    shd_arrays.append(day_array)



fig, axs = plt.subplots(nrows=1, ncols=2)
ax1 = axs[0]
ax2 = axs[1]
ax1.imshow(dem.values[0])
ax2.imshow(HS, cmap='gray')
