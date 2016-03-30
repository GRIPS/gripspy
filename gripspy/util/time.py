"""
Module for time conversion
"""
from __future__ import division, absolute_import, print_function

import numpy as np


__all__ = ['oeb2utc']


def oeb2utc(systime_array):
    """Converts 6-byte system time (AKA gondola time) to UTC during the flight.  Note that the
    detectors report a higher-precision "event" time that must first be divided by 10 to convert
    to system time.

    Parameters
    ----------
    systime_array : `~numpy.ndarray`
        Array of system times (normally `~numpy.int64`)

    Returns
    -------
    utc_array : `~numpy.ndarray`
        Array of UTC times `~numpy.datetime64`

    Notes
    -----
    This conversion function works as intended for only system times during the flight.
    The flight computer was restarted on 2016 Jan 25, which leads to a discontinuity in system
    times, and this discontinuity is taken into account by the function.
    """
    # Best-fit parameters provided by Pascal on 2016 Mar 30
    ref1_datetime64 = np.datetime64('2016-01-18T18:27:12Z')
    ref1_fit = (948121319866971.46, 0.99999868346236354)
    ref2_datetime64 = np.datetime64('2016-01-25T08:25:36Z')
    ref2_fit = (359386220206.81140, 0.99999774862264112)

    # System time converted from 100-ns steps to nominal 1-ns steps
    arr = (np.array(systime_array) * 100)

    # Convert the system time to UTC using the best-fit parameters
    utc_array = np.empty_like(arr, np.dtype('<M8[ns]'))
    utc_array[arr >= 6e14] = ref1_datetime64 +\
                             ((arr[arr >= 6e14] - ref1_fit[0]) / ref1_fit[1]).astype(np.int64).view(np.dtype('<m8[ns]'))
    utc_array[arr < 6e14] = ref2_datetime64 +\
                            ((arr[arr < 6e14] - ref2_fit[0]) / ref2_fit[1]).astype(np.int64).view(np.dtype('<m8[ns]'))

    return utc_array
