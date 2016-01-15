from __future__ import division, absolute_import, print_function

import os
import pickle
import gzip

import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt

from ..telemetry import parser_generator


__all__ = ['BGOData']


DIR = os.path.join(__file__, "..")


class BGOData(object):
    def __init__(self, telemetry_file=None, save_file=None):
        if telemetry_file is not None:
            self.filename = telemetry_file
            self.event_time = []
            self.channel = []
            self.level = []
            self.clock_source = []
            self.clock_synced = []

            count = 0

            with open(telemetry_file, 'rb') as f:
                pg = parser_generator(f, filter_systemid=0xB6, verbose=True)
                for p in pg:
                    count += len(p['event_time'])
                    self.event_time.append(p['event_time'])
                    self.channel.append(p['channel'])
                    self.level.append(p['level'])
                    self.clock_source.append(p['clock_source'])
                    self.clock_synced.append(p['clock_synced'])

            if count > 0:
                self.event_time = np.hstack(self.event_time)
                self.channel = np.hstack(self.channel)
                self.level = np.hstack(self.level)
                self.clock_source = np.hstack(self.clock_source)
                self.clock_synced = np.hstack(self.clock_synced)

                print("Total events: {0}".format(count))
            else:
                print("No events found")

        elif save_file is not None:
            with gzip.open(save_file, 'rb') as f:
                saved = pickle.load(f)
                self.filename = saved['filename']
                self.event_time = saved['event_time']
                self.channel = saved['channel']
                self.level = saved['level']
                self.clock_source = saved['clock_source']
                self.clock_synced = saved['clock_synced']
        else:
            raise RuntimeError("Either a telemetry file or a save file must be specified")

    def save(self, save_file=None):
        """Save the parsed data for future reloading.
        The data is stored in gzip-compressed binary pickle format.

        Parameters
        ----------
        save_file : str
            The name of the save file to create.  If none is provided, the default is the name of
            the telemetry file with the extension ".bgo.pgz" appended.

        """
        if save_file is None:
            save_file = self.filename + ".bgo.pgz"

        with gzip.open(save_file, 'wb') as f:
            pickle.dump({'filename' : self.filename,
                         'event_time' : self.event_time,
                         'channel' : self.channel,
                         'level' : self.level,
                         'clock_source' : self.clock_source,
                         'clock_synced' : self.clock_synced}, f, pickle.HIGHEST_PROTOCOL)

    @property
    def c(self):
        return self.channel

    @property
    def e(self):
        return self.event_time

    @property
    def l(self):
        return self.level

    @property
    def l0(self):
        return np.flatnonzero(self.level == 0)

    @property
    def l1(self):
        return np.flatnonzero(self.level == 1)

    @property
    def l2(self):
        return np.flatnonzero(self.level == 2)

    @property
    def l3(self):
        return np.flatnonzero(self.level == 3)
