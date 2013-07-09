'''
Generic regrid by arbitrary (possibly multidimensional) coords.

'''

import numpy as np

import iris.analysis

def stats_over_coord_grid(source_cube, grid_cube,
                          aggregator_list=[iris.analysis.MEAN]):
    """
    Grid data by coordinate value, without reference to a coordinate system.

    Args:
    * source_cube, grid_cube (:class:`iris.cube.Cube'):
        The source data cube.

    * grid_cube (:class:`iris.cube.Cube'):
        A cube whose coordinates define the desired target grid.
        Only the contiguous bounds of the cube's DimCoords are actually used.
        Each of these must have the same name as a coordinate of source_cube.

    * aggregator_list (sequence of :class:`iris.analysis.Aggregator`):
        Methods defining the output statistics, each of which produce a single
        value over the set of source points falling in each output cell.

    Return value:
        A new cube, which is a copy of the original data cube aggregated onto
        the grid defined by grid_cube.

    Note::
        All source cube coordinates which share any dimensions with regridded
        coordinates are lost.  Any coordinates mapped to other dimensions, and 
        scalar coordinates, are retained.
        ??? A cell method is added to describe the resampling of the data.

    """

def test_regrid_over_coords():
    pass


if __name__ == '__main__':
    test_regrid_over_coords()
