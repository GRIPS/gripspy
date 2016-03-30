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
    Also, the flight computer was restarted on 2016 Jan. 25, which leads to a discontinuity in
    system times.
    """
    # These reference points need to be updated
    ref1_systime = 14864585535293
    ref1_datetime64 = np.datetime64('2016-01-25T00:00:01.7021563Z')
    ref2_systime = 600373974638
    ref2_datetime64 = np.datetime64('2016-01-26T01:00:23.0193764Z')

    arr = (np.array(systime_array) * 100).astype(np.int64)

    arr[arr < 6e14] += (ref1_systime - ref2_systime) * 100 - (ref1_datetime64 - ref2_datetime64).astype(np.int64)
    arr -= ref1_systime * 100

    return ref1_datetime64 + arr.view(np.dtype('<m8[ns]'))
