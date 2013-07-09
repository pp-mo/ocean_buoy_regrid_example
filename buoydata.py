'''
Created on Jul 3, 2013

@author: itpp
'''

import math
import netCDF4 as nc
import numpy as np

import iris
from iris.fileformats.pp import EARTH_RADIUS

from regular_points import regular_points
from timed_block import TimedBlock

def from_file(nc_filepath):
    """
    Read in an ocean buoy profiles file.

    Args:
    * file (file or filepath):
        read from this.
        The file is expected to include the following:
            basic numberic indices :
                DATE_TIME
                N_PARAM
                N_PROF
                N_LEVELS
            key coordinate information:
                JULD(N_PROF)  --> datetime : station_time
                LATITUDE(N_PROF)  --> profile_lat
                LONGITUDE(N_PROF) --> profile_lon
            main data:
                DEPH_CORRECTED(N_PROF, N_LEVELS)  --> depth [m]
                PSAL_CORRECTED(N_PROF, N_LEVELS)  --> salinity [0.01]  ((percentage?))
                TEMP(N_PROF, N_LEVELS)  --> temperature [K]
                POTM_CORRECTED(N_PROF, N_LEVELS) --> potential_temp [K]

    Return:
        a cube for each of the main variables, with approriate coords.

    There is a lot this does *not* do.
    It ignores all the QC stuff for now, and much else.

    """
    with nc.Dataset(nc_filepath) as ds:
        # Get basic dimensions from file
        n_profiles = len(ds.dimensions['N_PROF'])
        n_levels = len(ds.dimensions['N_LEVELS'])
        # in practice, found that all of JULD == JULD_LOCATION
        # Don't really understand that now.  Use JULD for now ???

        # Get per-profile file variables  == time, lat, lon
        var_profile_juld = ds.variables['JULD']
        var_profile_lats = ds.variables['LATITUDE']
        var_profile_lons = ds.variables['LONGITUDE']
        for var in (var_profile_juld, var_profile_lats, var_profile_lons):
            assert var.dimensions == ('N_PROF',)
        profile_times_units = var_profile_juld.units
        profile_times = var_profile_juld[:]
        profile_lats = var_profile_lats[:]
        profile_lons = var_profile_lons[:]
        for data in (profile_times, profile_lats, profile_lons):
            assert data.shape == (n_profiles,)

        # get per-level main data
        var_depth = ds.variables['DEPH_CORRECTED']
        # TODO: "Depth" is essential, but others could be selected
        # programmatically?
        # E.G. vars[profile, level] with no 'QC' in name ??
#        basevar_names = ['PSAL_CORRECTED', 'TEMP', 'POTM']
        var_psal = ds.variables['PSAL_CORRECTED']
        var_temp = ds.variables['TEMP']
        var_potm = ds.variables['POTM_CORRECTED']
        for var in (var_depth, var_psal, var_temp, var_potm):
            assert var.dimensions == ('N_PROF', 'N_LEVELS')

        # Get the actual values + check dims
        depth_array = np.ma.masked_equal(var_depth[:], var_depth._fillvalue)
#        temp_array = var_temp[:]
#        psal_array = var_psal[:]
        potm_array = np.ma.masked_equal(var_potm[:], var_potm._fillvalue)
#        for data in (depth_array, psal_array, temp_array, potm_array):
        for data in (depth_array, potm_array):
            assert data.shape == (n_profiles, n_levels)

        # Return this basic info as CUBES[n_profiles, n_levels], but with
        # with DEPTH as an auxiliary coord on all of them.
        depth_coord = iris.coords.AuxCoord(depth_array,
                                           standard_name='depth',
                                           units='m')

        time_coord = iris.coords.AuxCoord(profile_times,
                                          standard_name='time',
                                          units=profile_times_units)

        cs_basic_latlon = iris.coord_systems.GeogCS(EARTH_RADIUS)
        lons_coord = iris.coords.AuxCoord(profile_lons,
                                          standard_name='longitude',
                                          units='degrees',
                                          coord_system=cs_basic_latlon)

        lats_coord = iris.coords.AuxCoord(profile_lats,
                                          standard_name='latitude',
                                          units='degrees',
                                          coord_system=cs_basic_latlon)


        # Hack the units for now...
        potm_cube = iris.cube.Cube(potm_array,
                                   standard_name = 'sea_water_potential_temperature',
                                   units='degrees celsius')
        potm_cube.add_aux_coord(depth_coord, (0,1))
        potm_cube.add_aux_coord(time_coord, (0,))
        potm_cube.add_aux_coord(lons_coord, (0,))
        potm_cube.add_aux_coord(lats_coord, (0,))
        return potm_cube


def test_from_file():
    test_filepath = '/data/local/dataZoo/NetCDF/oceanObs/EN3_v2a_Profiles_195001.nc'
    cube = from_file(test_filepath)
    print cube
    print 'minmax = ',[np.min(cube.data), np.max(cube.data)]

    lats = cube.coord('latitude').points
    lons = cube.coord('longitude').points
    depths = cube.coord('depth').points

    n_profiles, n_levels = depths.shape

    lats_range = np.min(lats), np.max(lats)
    lons_range = np.min(lons), np.max(lons)
    depths_range = np.min(depths), np.max(depths)

    lon_levels = regular_points(start_value=lons_range[0],
                                stop_value=lons_range[1],
                                step_size=2.0,
                                allow_range_extend=True)
    lat_levels = regular_points(start_value=lats_range[0],
                                stop_value=lats_range[1],
                                step_size=2.0,
                                allow_range_extend=True)
    depth_levels = np.array([0.0, 10.0, 50.0, 150.0, 300.0])
#    depth_levels = np.array([0.0, 300.0])
    n_lat_cells = len(lat_levels) - 1
    n_lon_cells = len(lon_levels) - 1
    n_depth_cells = len(depth_levels) - 1
    data_shape = (n_depth_cells, n_lat_cells, n_lon_cells)
    print
    print 'Result dims = ', data_shape
    a_counts = np.zeros(data_shape, dtype=np.int)
    a_means = np.ma.zeros(data_shape)
    a_means[:] = np.ma.masked
    for id in range(n_depth_cells):
#    for id in range(0, 1):
        print 'id = {}/{}'.format(id, n_depth_cells)
        d0, d1 = depth_levels[id:id+2]
        for iy in range(n_lat_cells):
#        for iy in range(10,27):
            print '  iy = {}/{}'.format(iy, n_lat_cells)
            y0, y1 = lat_levels[iy:iy+2]
            for ix in range(n_lon_cells):
                x0, x1 = lon_levels[ix:ix+2]
                indexes = np.where(
                    np.logical_and(
                        np.logical_and(depths >= d0, depths <= d1),
                        np.logical_and(
                            np.logical_and(lats >= y0, lats <= y1),
                            np.logical_and(lons >= x0, lons <= x1)).reshape(
                                (n_profiles, 1))))
                n_points = len(indexes[0])
                a_counts[id, iy, ix] = n_points
                average = np.ma.average(cube.data[indexes]) \
                    if n_points else np.ma.masked
                a_means[id, iy, ix] = average
    np.set_printoptions(precision=3, threshold=100e6, linewidth=180)
    print
    print 'Counts..'
    print a_counts
    print
    print 'Means..'
    print a_means


# Example result:
#    sea_water_potential_temperature / (degrees celsius) (*ANONYMOUS*: 2581; *ANONYMOUS*: 55)
#         Auxiliary coordinates:
#              latitude                                              x                  -
#              longitude                                             x                  -
#              time                                                  x                  -
#              depth                                                 x                  x

def testdata_small():
    """Make 2d data, lats, lons and lat_levels, lon_levels test arrays."""
    do_tiny = False
    if do_tiny:
        lats = np.array([10, 30, 20, 50, 30, 60, 15])
        lons = np.array([50, 40, 20, 50, 22, 99, 30])
        data = np.array([33, 77, 22, 20, 50, 99, 22])
        lat_levels = np.array([0, 25, 50])
        lon_levels = np.array([25, 30, 40, 60])
    else:
        lats = np.array([[10, 20, 30, 20],
                         [20, 50, 20, 60],
                         [15, 20, 30, 50]])
        lons = np.array([[40, 40, 40, 40],
                         [10, 20, 20, 20],
                         [50, 50, 50, 50]])
        data = np.array([[33, 77, 55, 22],
                         [77, 20, 22, 20],
                         [50, 99, 50, 22]])
        lat_levels = np.array([0, 10, 20, 30, 40, 50])
        lon_levels = np.array([0, 10, 20, 30, 40, 50, 60])
    return data, lats, lons, lat_levels, lon_levels


def testdata_random(nx=4, ny=3, nlats=6, nlons=7):
    """Make 2d data, lats, lons and lat_levels, lon_levels test arrays."""
    lats = np.random.uniform(5.0, 85.0, (ny,nx))
    lons = np.random.uniform(5.0, 85.0, (ny,nx))
    data = np.random.uniform(0.0, 99.0, (ny,nx))
    lat_levels = sorted(np.random.uniform(0.0, 90.0, nlats))
    lon_levels = sorted(np.random.uniform(0.0, 90.0, nlons))
    return data, lats, lons, lat_levels, lon_levels


def regrid_obvious(data, lats, lons, lat_levels, lon_levels):
    """
    Regrid data into bins over lats and lons into bins.

    For now, return counts and means.
    """
    n_lat_cells = len(lat_levels) - 1
    n_lon_cells = len(lon_levels) - 1
    data_shape = (n_lat_cells, n_lon_cells)

    # For now, make counts + averages (with mask allowance)
    a_counts = np.zeros(data_shape, dtype=np.int)
    a_means = np.ma.zeros(data_shape)
    a_means[:] = np.ma.masked
    for iy in range(n_lat_cells):
        y0, y1 = lat_levels[iy:iy+2]
        for ix in range(n_lon_cells):
            x0, x1 = lon_levels[ix:ix+2]
#            indexes = np.where(np.logical_and(
#                np.logical_and(lats >= y0, lats < y1),
#                np.logical_and(lons >= x0, lons < x1)))
            indexes = np.logical_and(
                np.logical_and(lats >= y0, lats < y1),
                np.logical_and(lons >= x0, lons < x1))
            a_counts[iy, ix] = len(np.where(indexes)[0])
            a_means[iy, ix] = np.average(data[indexes])
    return a_counts, a_means


def regrid_better(data, lats, lons, lat_levels, lon_levels):
    # Construct shape of index arrays = [n-dims, dim1, dim2..]
    n_lat_cells = len(lat_levels) - 1
    n_lon_cells = len(lon_levels) - 1
#    lat_cell_compares = lats.reshape([1] + list(lats.shape)) \
#        >= lat_levels.reshape(list(lat_levels.shape) + [1] * lats.ndim)
#    lon_cell_compares = lons.reshape([1] + list(lons.shape)) \
#        >= lon_levels.reshape(list(lon_levels.shape) + [1] * lats.ndim)
    lat_cell_compares = np.less_equal.outer(lat_levels, lats)
    lon_cell_compares = np.less_equal.outer(lon_levels, lons)
    lat_cell_indices = np.sum(lat_cell_compares, axis=0)
    lon_cell_indices = np.sum(lon_cell_compares, axis=0)
#    del lat_cell_compares
#    del lon_cell_compares


    lat_cell_ids = np.arange(1, n_lat_cells+1).reshape([1] * data.ndim + [n_lat_cells, 1])
    lon_cell_ids = np.arange(1, n_lon_cells+1).reshape([1] * data.ndim + [1, n_lon_cells])
    all_lat_cells = lat_cell_indices.reshape(list(data.shape) + [1, 1])
    all_lon_cells = lon_cell_indices.reshape(list(data.shape) + [1, 1])
    lat_cell_matches = (all_lat_cells == lat_cell_ids)
    lon_cell_matches = (all_lon_cells == lon_cell_ids)
    matches = (lat_cell_matches & lon_cell_matches)

    all_vals = np.ma.zeros(list(data.shape) + [n_lat_cells, n_lon_cells])
    all_vals[:] = data.reshape(list(data.shape) + [1, 1])
    all_vals.mask = ~matches
    all_vals = all_vals.reshape((data.size, n_lat_cells, n_lon_cells))
    counts = np.ma.count(all_vals, axis=0)
    means = np.ma.average(all_vals, axis=0)
#    mins = np.ma.min(all_vals, axis=0)
#    maxs = np.ma.max(all_vals, axis=0)
    return counts, means


def test_regrid_obvious(testdata=None):
    if testdata is None:
        testdata = testdata_small()
    a_counts, a_means = regrid_obvious(*testdata)
    print
    print 'Counts..'
    print a_counts
    print
    print 'Means..'
    print a_means


def test_regrid_better(testdata=None):
    if testdata is None:
        testdata = testdata_small()
    a_counts, a_means = regrid_better(*testdata)
    print
    print 'Counts..'
    print a_counts
    print
    print 'Means..'
    print a_means


def test_times(do_small=False, test_dims = (170, 150, 15, 12)):
    if do_small:
        testdata = [None]
    else:
        nx, ny, nlats, nlons = test_dims
        testdata = testdata_random(nx, ny, nlats, nlons)
    print 'Testing simplistic solution to small case...'
    with TimedBlock() as t:
        test_regrid_obvious(testdata)
    print '  .. time taken = ', t.seconds()

    print
    print 'Testing clever solution to small case...'
    with TimedBlock() as t:
        test_regrid_better(testdata)
    print '  .. time taken = ', t.seconds()


if __name__ == '__main__':
#    test_from_file()
#    test_regrid_small_obvious()
#    test_regrid_small_better()
#    test_regrid_example()
    test_times()

