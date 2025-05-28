import numpy as np
import rasterio as rio
from pathlib import Path
import timeit

import insolpy
import config

# These are HUC12 sized basins
dem_dir = Path(r'C:\Users\CNB968\OneDrive - MT\GitHub\insolpy\Examples\trhuc12_tiffs')
fls = list(dem_dir.glob('*.tif'))
test_dem_pth = fls[0]

# UY test, approximately HUC8 size basin
HUC8_dir = Path(r'D:\ArcGIS_Projects\Yellowstone\Upper Yellowstone\prms\MT_hydro_SRTM_30m_clipped.tif')

with rio.open(test_dem_pth) as src:
    huc12arr = src.read(1)
    huc12res = src.res[0]

with rio.open(HUC8_dir) as src:
    huc8arr = src.read(1)
    huc8res = src.res[0]

sunvec = config.R_normalvector.normalvector(58, 200)

cg = insolpy.cgrad(huc12arr, dlx=huc12res)

print(f"Test terrain correction functions for run time. These tests are performed on a single HUC12 watershed, array sized ({huc12arr.shape[0], huc12arr.shape[1]}).")
print("These test runs use just a single sun position or sun vector.")
print(f"insolpy.cgrad(): {min(timeit.Timer(lambda: insolpy.cgrad(huc12arr, dlx=huc12res)).repeat(5, 1))} sec; minimum of 5 runs, 1 loop each.")
print(f"insolpy.hillshading(): {min(timeit.Timer(lambda: insolpy.hillshading(cg, sunvec)).repeat(5, 1))} sec; minimum of 5 runs, 1 loop each.")
print(f"insolpy.doshade(): {min(timeit.Timer(lambda: insolpy.doshade(huc12arr, sunvec, huc12res)).repeat(5, 1))} sec; minimum of 5 runs, 1 loop each.")
print("This is to test the accelerated doshade() function - fast_doshade() which should have better performance because of numba runtime compiling.")
print(f"fast_doshade(): {min(timeit.Timer(lambda: insolpy.fast_doshade(huc12arr.astype(np.float64), np.array([200, 58]), huc12res)).repeat(5, 1))} sec; minimum of 5 runs, 1 loop each.")

print("doshade is clearly the performance issue because of nested looping. The next tests perform shade mapping for 500 sun positions on a single HUC12.")
randaz = np.random.uniform(90, 270, 500)
randzen = np.random.uniform(10, 80, 500)

def many_sunpos_slow(azarr, zenarr):
    arrays = []
    for i in range(len(randaz)):
        out = insolpy.doshade(huc12arr, sunvec, huc12res)
        arrays.append(out)

    return arrays

def many_sunpos_fast(azarr, zenarr):
    arrays = []
    for i in range(len(randaz)):
        out = insolpy.fast_doshade(huc12arr.astype(np.float64), np.array([200, 58]), huc12res)
        arrays.append(out)

    return arrays

def many_sunpos_fasthuc8(azarr, zenarr):
    arrays = []
    for i in range(len(randaz)):
        out = insolpy.fast_doshade(huc8arr.astype(np.float64), np.array([200, 58]), huc8res)
        arrays.append(out)

    return arrays

print(f"doshade() with 500 sun positions: {min(timeit.Timer(lambda: many_sunpos_slow(randaz, randzen)).repeat(5, 1))} sec; minimum of 5 runs, 1 loop each.")
print(f"fast_doshade() with 500 sun positions: {min(timeit.Timer(lambda: many_sunpos_fast(randaz, randzen)).repeat(5, 1))} sec; minimum of 5 runs, 1 loop each.")

print("We know that the fast version of doshade() has a significant performance boost. Here is a further test of it on an ~HUC8 size basin.")
print(f"fast_doshade(): {min(timeit.Timer(lambda: insolpy.fast_doshade(huc8arr.astype(np.float64), np.array([200, 58]), huc8res)).repeat(5, 1))} sec; minimum of 5 runs, 1 loop each.")
print("And, a test of the HUC8 size for 500 sun positions.")
print(f"fast_doshade() with 500 sun positions: {min(timeit.Timer(lambda: many_sunpos_fasthuc8(randaz, randzen)).repeat(5, 1))} sec; minimum of 5 runs, 1 loop each.")
