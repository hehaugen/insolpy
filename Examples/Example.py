import rioxarray
import numpy as np
import xarray as xr
from pathlib import Path
from multiprocessing import Pool

from src import dailyshade


# first get some DEM data for a HUC12 cathment
#huc12 = '100901010101'
#geom = pynhd.WaterData('wbd12').byid('huc12', huc12).geometry[0]
#dem = py3dep.get_dem(geom, 30)
#dem_proj = dem.rio.reproject(5071)

def shade_helper(dem_pth):
    dem = rioxarray.open_rasterio(dem_pth)
    huc12 = dem_pth.name.split('_')[0]
    dem_bbox = dem.rio.bounds()
    lon = (dem_bbox[0] + dem_bbox[2]) / 2
    lat = (dem_bbox[1] + dem_bbox[3]) / 2
    dem_proj = dem.rio.reproject(5071)

    tmzn = -7
    resltn = dem_proj.rio.resolution()[0]

    Result = dailyshade(dem_proj.values[0], resltn, lat, lon, tmzn, '2000-01-01', '2000-12-31')

    R_dset = xr.Dataset(
        {
            "shd_f": (['y', 'x', 'time'], np.dstack(Result[0]), {'standard_name': 'shade_factor', 'units': 'None'}),
            "elevation": (['y', 'x'], dem_proj.sel(band=1).data, {'standard_name': 'elevation', 'units': 'm'})
        },
        coords={
            "y": (['y'], dem_proj.y.data, {'standard_name': 'projected_y_coordinate', 'units': 'm'}),
            "x": (['x'], dem_proj.x.data, {'standard_name': 'projected_x_coordinate', 'units': 'm'}),
            "time": (['time'], Result[1])
        }
    )

    R_dset.rio.write_crs(dem_proj.rio.crs, inplace=True)
    R_dset.to_netcdf('{0}.nc'.format(huc12))
    print("Finished DEM {0}".format(dem_pth.name))


if __name__ == '__main__':
    dir = Path(r'C:\Users\CNB968\OneDrive - MT\GitHub\insolpy\Examples\trhuc12_tiffs')
    fls = list(dir.glob('*.tif'))
    print("{0} DEM paths loaded...".format(len(fls)))

    #shade_helper(fls[0])

    with Pool(20) as pool:
        pool.map(shade_helper, fls)
    print("Finished all Processes.")
