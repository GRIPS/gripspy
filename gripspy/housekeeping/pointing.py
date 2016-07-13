"""
Module for analyzing pointing data
"""
from __future__ import division, absolute_import, print_function

import os
import pickle
import gzip

import numpy as np
import astropy.units as u
from astropy.coordinates import Angle

from ..telemetry import parser_generator


__all__ = ['PointingData']


DIR = os.path.join(__file__, "..")


class PointingData(object):
    """Class for analyzing pointing data

    Parameters
    ----------
    telemetry_file : str
        The name of the telemetry file to analyze.  If None is specified, a save file must be specified.
    save_file : str
        The name of a save file from a telemetry file that was previously parsed.

    Notes
    -----
    This implementation is still incomplete!
    """
    def __init__(self, telemetry_file=None, save_file=None):
        if telemetry_file is not None:
            self.filename = telemetry_file

            self.systime = []
            self.v_x = []
            self.v_y = []
            self.v_sum = []

            self.elevation_ae = []
            self.elevation_ie = []
            self.solar_elevation = []

            count = 0

            with open(telemetry_file, 'rb') as f:
                pg = parser_generator(f, filter_systemid=0x03, filter_tmtype=0x03, verbose=True)
                for p in pg:
                    count += 1
                    self.systime.append(p['systime'])
                    self.v_x.append(p['v_x'])
                    self.v_y.append(p['v_y'])
                    self.v_sum.append(p['v_sum'])

                    self.elevation_ae.append(p['elevation_ae'])
                    self.elevation_ie.append(p['elevation_ie'])
                    self.solar_elevation.append(p['solar_elevation'])

            if count > 0:
                self.systime = np.hstack(self.systime)
                self.v_x = np.hstack(self.v_x)
                self.v_y = np.hstack(self.v_y)
                self.v_sum = np.hstack(self.v_sum)

                self.elevation_ae = Angle(self.elevation_ae, 'deg')
                self.elevation_ie = Angle(self.elevation_ie, 'deg')
                self.solar_elevation = Angle(self.solar_elevation, 'deg')

                print("Total packets: {0}".format(count))
            else:
                print("No packets found")

        elif save_file is not None:
            with gzip.open(save_file, 'rb') as f:
                saved = pickle.load(f)
                self.filename = saved['filename']
                self.systime = saved['systime']
                self.v_x = saved['v_x']
                self.v_y = saved['v_y']
                self.v_sum = saved['v_sum']
                self.elevation_ae = saved['elevation_ae']
                self.elevation_ie = saved['elevation_ie']
                self.solar_elevation = saved['solar_elevation']
        else:
            raise RuntimeError("Either a telemetry file or a save file must be specified")

    def save(self, save_file=None):
        """Save the parsed data for future reloading.
        The data is stored in gzip-compressed binary pickle format.

        Parameters
        ----------
        save_file : str
            The name of the save file to create.  If none is provided, the default is the name of
            the telemetry file with the extension ".pointing.pgz" appended.

        """
        if save_file is None:
            save_file = self.filename + ".pointing.pgz"

        with gzip.open(save_file, 'wb') as f:
            pickle.dump({'filename' : self.filename,
                         'systime' : self.systime,
                         'v_x' : self.v_x,
                         'v_y' : self.v_y,
                         'v_sum' : self.v_sum,
                         'elevation_ae' : self.elevation_ae,
                         'elevation_ie' : self.elevation_ie,
                         'solar_elevation' : self.solar_elevation}, f, pickle.HIGHEST_PROTOCOL)
