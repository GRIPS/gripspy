"""
Module for analyzing GPS and pressure data from the SIP
"""
from __future__ import division, absolute_import, print_function

import os
import pickle
import gzip

import numpy as np
import astropy.units as u
from astropy.units import cds
u.add_enabled_units([cds.mbar])
from astropy.coordinates import Latitude, Longitude

from ..telemetry import parser_generator


__all__ = ['GPSData']


DIR = os.path.join(__file__, "..")


class GPSData(object):
    """Class for analyzing GPS and pressure data from the SIP

    Parameters
    ----------
    telemetry_file : str
        The name of the telemetry file to analyze.  If None is specified, a save file must be specified.
    save_file : str
        The name of a save file from a telemetry file that was previously parsed.

    Notes
    -----
    Trusts the "user" GPS information
    Averages the SIP1 and SIP2 pressure information
    """
    def __init__(self, telemetry_file=None, save_file=None):
        if telemetry_file is not None:
            self.filename = telemetry_file

            self.systime = []
            self.latitude = []
            self.longitude = []
            self.altitude = []

            self.utc_time = []

            pressure_low = []
            pressure_mid = []
            pressure_high = []

            count = 0

            print("Parsing {0}".format(telemetry_file))
            with open(telemetry_file, 'rb') as f:
                pg = parser_generator(f, filter_systemid=0x05, filter_tmtype=0x02, verbose=True)
                for p in pg:
                    count += 1
                    self.systime.append(p['systime'])
                    self.latitude.append(p['user_latitude'])
                    self.longitude.append(p['user_longitude'])
                    self.altitude.append(p['user_altitude'])

                    # Convert time to UTC
                    self.utc_time.append(np.datetime64('1980-01-06T00:00:00Z') +\
                                         np.timedelta64(p['user_week'], 'W') +\
                                         np.timedelta64(int((p['user_timeofweek'] - p['user_offset']) * 1e6), 'us'))

                    pressure_low.append((p['sip1_pressure_low'] + p['sip2_pressure_low']) / 2)
                    pressure_mid.append((p['sip1_pressure_mid'] + p['sip2_pressure_mid']) / 2)
                    pressure_high.append((p['sip1_pressure_high'] + p['sip2_pressure_high']) / 2)

            if count > 0:
                self.systime = np.hstack(self.systime)
                self.latitude = Latitude(self.latitude, 'deg')
                self.longitude = Longitude(self.longitude, 'deg', wrap_angle=180 * u.deg)
                self.altitude = u.Quantity(self.altitude, 'm').to('km')

                self.utc_time = np.hstack(self.utc_time)

                pressure_low = np.hstack(pressure_low)
                pressure_mid = np.hstack(pressure_mid)
                pressure_high = np.hstack(pressure_high)

                # Convert pressure to millibars
                self.pressure = pressure_low * 0.327 - 10.76
                use_mid = pressure_mid < 3165
                self.pressure[use_mid] = pressure_mid[use_mid] * 0.032 - 1.29
                use_high = pressure_high < 3383
                self.pressure[use_high] = pressure_high[use_high] * 0.003 - 0.149

                self.pressure = u.Quantity(self.pressure, 'mbar')

                print("Total packets: {0}".format(count))
            else:
                print("No packets found")

        elif save_file is not None:
            print("Restoring {0}".format(save_file))
            with gzip.open(save_file, 'rb') as f:
                saved = pickle.load(f)
                self.filename = saved['filename']
                self.systime = saved['systime']
                self.latitude = saved['latitude']
                self.longitude = saved['longitude']
                self.altitude = saved['altitude']
                self.utc_time = saved['utc_time']
                self.pressure = saved['pressure']
        else:
            raise RuntimeError("Either a telemetry file or a save file must be specified")

    def save(self, save_file=None):
        """Save the parsed data for future reloading.
        The data is stored in gzip-compressed binary pickle format.

        Parameters
        ----------
        save_file : str
            The name of the save file to create.  If none is provided, the default is the name of
            the telemetry file with the extension ".gps.pgz" appended.

        """
        if save_file is None:
            save_file = self.filename + ".gps.pgz"

        with gzip.open(save_file, 'wb') as f:
            pickle.dump({'filename' : self.filename,
                         'systime' : self.systime,
                         'latitude' : self.latitude,
                         'longitude' : self.longitude,
                         'altitude' : self.altitude,
                         'utc_time' : self.utc_time,
                         'pressure' : self.pressure}, f, pickle.HIGHEST_PROTOCOL)
