import rioxarray
import geopandas as gpd
import numpy as np
from pathlib import Path
from multiprocessing import Pool
import time

from insolpy import dailyshade_geometry

rasterdir = Path(r'C:\Users\CNB968\OneDrive - MT\GitHub\insolpy\Examples\Tongue_R_30m.tif')
geomdir = Path(r'D:\ArcGIS_Projects\Yellowstone\Tongue River\Vector\HWWY_reservoirs.shp')
G = gpd.read_file(geomdir)
Gproj = G.to_crs(5070)
nhdIDs = Gproj['Permanent_'].tolist()

def shade_helper(nhdid):
    dem = rioxarray.open_rasterio(rasterdir)
    geom = Gproj.loc[Gproj['Permanent_'] == nhdid,:]
    geom.loc[:, 'geometry'] = geom.centroid
    tmzn = -7

    result = dailyshade_geometry(dem, geom, tmzn, '2000-06-01', '2000-06-30')

    result.to_csv('test{0}.csv'.format(nhdid))
    print("Finished geometry.")


if __name__ == '__main__':
    strt_time = time.time()
    shade_helper(nhdIDs[0])
    end_time = time.time()

    # with Pool(20) as pool:
    #     pool.map(shade_helper, nhdIDs)
    # print("Finished all Processes.")
    print(f"Elapsed time: {end_time - strt_time} seconds.")