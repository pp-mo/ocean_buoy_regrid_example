'''
Created on Jul 3, 2013

@author: itpp
'''

import math
import numpy as np

def regular_steps_params(start_value, stop_value=None,
                         n_points=None, step_size=None,
                         allow_range_extend=False):
    """
    Process the args (start_value, stop_value, n_points, step_size) as
    required to define regularly spaced value points in the range
    (start_value, stop_value), filling in or altering as required to ensure
    that stop_value = start_value + (n_points - 1) * step_size.

    Args:
    * start_value, stop_value (number):
        The outer ends of the resulting value range.
        stop_value is not required if both n_points and step_size are given.

    Kwargs:
    * n_points (int):
        Number of points required.  Must be at least 2.

    * step_size (float):
        Distance between adjacent points.
        If this has the opposite sign (i.e. moves from stop toward start), then
        the result will have n_points == 1.

    * allow_range_extend (bool):
        If set, when n_points is calculated from step_size, the last point is
        at-or-beyond stop_value, rather than at-or-before (the default).

    Returns:
        (start_value, stop_value, n_points, step_size)
        Made consistent so that:
            stop_value = start_value + (n_points - 1) * step_size

    Note::
        Either n_points or step_size *must* be specified.
        If step_size is not set, the end points are always exactly start_value
        and stop_value.
        If step_size is set, but not n_points, then the last point is at or 
        before stop_value.
        If both n_points and step_size are given, then stop_value is entirely
        ignored and may be omitted.


    Note::
        Either n_points or step_size *must* be specified.
        If step_size is not set, the end points are always exactly start_value
        and stop_value.
        If step_size is set, but not n_points, then n_points is chosen so that
        the last point is at or before the stop_point.
        If both n_points and step_size are given, then stop_value is reset to
        match.

    Examples:
        regular_points(0.0, 2.0, 1)
          --> [0.0, 1.0]

        regular_points(0.0, 2.0, 2)
          --> [0.0, 0.5, 1.0]

        regular_points(0.0, 1.0, step_size=0.3)
          --> [0.0, 0.3, 0.6, 0.9]

        regular_points(0.6, step_size=0.6, n_points=3)
          --> [0.6, 1.2, 1.8]

    """
    if n_points is not None and n_points < 2:
        raise ValueError('n_points must be at least 2.')
    # Process inputs into the standard form : start, stop, n
    if step_size is None:
        # no width : should be (start, stop, n)
        if n_points is None:
            raise ValueError('Must specify one of n_points or step_size.')
        if stop_value is None:
            raise ValueError('Must specify one of step_size or stop_value.')
        step_size = (stop_value - start_value) / (n_points - 1)
    else:
        # have width : can be (start, stop, width) or (start, width, n)
        if n_points is None:
            # no n, should be (start, stop, width)
            if stop_value is None:
                raise ValueError('Must specify one of n_points or stop_value.')
            # Choose number of points
            if allow_range_extend:
                n_steps = max(0, math.ceil((stop_value - start_value) / step_size))
            else:
                n_steps = max(0, math.floor((stop_value - start_value) / step_size))
            n_points = n_steps + 1
        # from (start, width, n) : ignore stop
        stop_value = start_value + (n_points - 1) * step_size
    return start_value, stop_value, n_points, step_size

def regular_points(start_value, stop_value=None,
                   n_points=None, step_size=None,
                   allow_range_extend=False):
    """
    Construct simple bounds and points arrays over a value range.

    The bounds cover the given range contiguously, and the points are at a set
    fraction from each lower bound.

    Args:
    * start_value, stop_value (number):
        The outer ends of the resulting bounds.
        stop_value is not required if both n_points and step_size are given.

    Kwargs:
    * n_points (int):
        Number of points to return.

    * step_size (float):
        Distance between adjacent points.
        If this has the opposite (i.e. moves from stop toward start), the
        result will be only one point.

    * allow_range_extend (bool):
        If set, when n_points is calculated from step_size, the last point is
        at-or-beyond stop_value, rather than at-or-before (the default).

    Returns:
        points[N]

    Note::
        Either n_points or step_size *must* be specified.
        If step_size is not set, the end points are always exactly start_value
        and stop_value.
        If step_size is set, but not n_points, then the last point is at or 
        before stop_value.
        If both n_points and step_size are given, then stop_value is entirely
        ignored and may be omitted.

    Examples:
        regular_points(0.0, 2.0, 1)
          --> [0.0, 1.0]

        regular_points(0.0, 2.0, 2)
          --> [0.0, 0.5, 1.0]

        regular_points(0.0, 1.0, step_size=0.3)
          --> [0.0, 0.3, 0.6, 0.9]

        regular_points(0.6, step_size=0.6, n_points=3)
          --> [0.6, 1.2, 1.8]

    """
    # Process inputs into the standard form : start, stop, n
    start_value, stop_value, n_points, step_size = \
        regular_steps_params(start_value, stop_value,
                             n_points, step_size,
                             allow_range_extend=allow_range_extend)
    points = np.linspace(start_value, stop_value, n_points, endpoint=True)
    return points

def regular_points_and_bounds(start_value, stop_value=None,
                              n_cells=None, cell_width=None,
                              clip_to_range=False,
                              points_position=0.5):
    """
    Construct simple bounds and points arrays over a value range.

    The bounds cover the given range contiguously, and the points are at a set
    fraction from each lower bound.

    Args:
    * start_value, stop_value (number):
        The outer ends of the resulting bounds.

    Kwargs:
    * n_cells (int):
        If set, the range is divided into this many equal cells.

    * cell_width (float):
        Width of each cell.  The last may extend beyond the stop_value.

    * points_position (float):
        Fractional position of point within each cell.

    * clip_to_range (bool):
        If set, then when n_cells is calculated from cell_width (instead of
        being given), then the last cell will always lie entirely inside the
        range (start_value, stop_value).  Thus, values near the end of the
        range will not be in any cell.
        Otherwise (the default), the last cell may cover values beyond the
        specified range, but all values in the range will lie in some cell.

    Returns:
        points[N], bounds[N,2]

    Note::
        Either n_steps or cell_width *must* be specified.
        If both n_steps and cell_width are given, then stop_value is ignored
        and may be omitted.

    Examples:
        pts, bounds = regular_points_and_bounds(0.0, 2.0, 2)
          --> [0.5, 1.5], [[0, 1], [1, 2]]

        pts, bounds = regular_points_and_bounds(0.0, 1.0, cell_width=0.4)
          --> [0.2, 0.6, 1.0], [[0, 0.4, 0.8], [0.4, 0.8, 1.2]]

        pts, bounds = regular_points_and_bounds(3.0, cell_width=0.6, n_cells=3)
          --> [3.3, 3.9, 4.5], [[3.0, 3.6, 4.2], [3.6, 4.2, 4.8]]

    """
    allow_range_extend = not clip_to_range
    n_points = n_cells + 1 if n_cells is not None else None
    start_value, stop_value, n_points, cell_width = \
        regular_steps_params(start_value=start_value,
                                    stop_value=stop_value,
                                    n_points=n_points,
                                    step_size=cell_width,
                                    allow_range_extend=allow_range_extend)
    cell_edges = np.linspace(start_value, stop_value, n_points, endpoint=True)
    bounds = np.array([cell_edges[:-1], cell_edges[1:]])
    points = bounds[0] + cell_width * points_position
    return points, bounds

def regrid_onto_coords(coords, coord_value_divisions):
    for coord in coords:
        pass

def test_from_file():
    test_filepath = '/data/local/dataZoo/NetCDF/oceanObs/EN3_v2a_Profiles_195001.nc'
    cube = from_file(test_filepath)
    print cube
    print 'minmax = ',[np.min(cube.data), np.max(cube.data)]

def test_regular_points():
    def test_pts(expected_result, *args, **kwargs):
        points = regular_points(*args, **kwargs)
        np.testing.assert_allclose(points, expected_result)

    test_pts([0.0, 1.0],
             0.0, 1.0, 2)

    test_pts([0.0, 0.5, 1.0],
             0.0, 1.0, 3)

    test_pts([0.0, 0.3, 0.6, 0.9],
             0.0, 1.0, step_size=0.3)

    test_pts([0.6, 1.2, 1.8],
             0.6, step_size=0.6, n_points=3)

    # test inversion cases
    test_pts([1.0, 0.5, 0.0],
             1.0, 0.0, 3)
    test_pts([1.0, 0.7, 0.4, 0.1],
             1.0, 0.0, step_size=-0.3)
    test_pts([1.0],
             1.0, 0.0, step_size=0.3)
    test_pts([0.6, 0.0, -0.6],
             0.6, step_size=-0.6, n_points=3)

    print 'regular_points ok.'


def test_regular_points_and_bounds():
    pts, bds = regular_points_and_bounds(0.0, 2.0, 2)
    np.testing.assert_allclose(pts, [0.5, 1.5])
    np.testing.assert_allclose(bds, [[0, 1], [1, 2]])

    pts, bds = regular_points_and_bounds(0.0, 1.0, cell_width=0.4)
    np.testing.assert_allclose(pts, [0.2, 0.6, 1.0])
    np.testing.assert_allclose(bds, [[0, 0.4, 0.8], [0.4, 0.8, 1.2]])

    pts, bds = regular_points_and_bounds(0.0, 1.0, cell_width=0.4, 
                                         clip_to_range=True)
    np.testing.assert_allclose(pts, [0.2, 0.6])
    np.testing.assert_allclose(bds, [[0, 0.4], [0.4, 0.8]])

    pts, bds = regular_points_and_bounds(3.0, cell_width=0.6, n_cells=3)
    np.testing.assert_allclose(pts, [3.3, 3.9, 4.5])
    np.testing.assert_allclose(bds, [[3.0, 3.6, 4.2], [3.6, 4.2, 4.8]])

    print 'regular_points_and_bounds ok.'

if __name__ == '__main__':
    test_regular_points()
    test_regular_points_and_bounds()
