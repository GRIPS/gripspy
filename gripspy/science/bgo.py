from __future__ import division, absolute_import, print_function

import os
import pickle
import gzip

import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt

from ..telemetry import parser_generator


__all__ = ['BGOEventData', 'BGOCounterData']


DIR = os.path.join(__file__, "..")


class BGOEventData(object):
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
                pg = parser_generator(f, filter_systemid=0xB6, filter_tmtype=0x82, verbose=True)
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


class BGOCounterData(object):
    def __init__(self, telemetry_file=None, save_file=None):
        if telemetry_file is not None:
            self.filename = telemetry_file
            self.counter_time = []
            self.total_livetime = []
            self.channel_livetime = []
            self.channel_count = []
            self.veto_count = []

            count = 0

            with open(telemetry_file, 'rb') as f:
                pg = parser_generator(f, filter_systemid=0xB6, filter_tmtype=0x81, verbose=True)
                for p in pg:
                    count += 1
                    self.counter_time.append(p['counter_time'])
                    self.total_livetime.append(p['total_livetime'])
                    self.channel_livetime.append(p['channel_livetime'])
                    self.channel_count.append(p['channel_count'])
                    self.veto_count.append(p['veto_count'])

            if count > 0:
                self.counter_time = np.hstack(self.counter_time)
                self.total_livetime = np.hstack(self.total_livetime)
                self.channel_livetime = np.vstack(self.channel_livetime)
                self.channel_count = np.dstack(self.channel_count).transpose((2, 0, 1))
                self.veto_count = np.hstack(self.veto_count)

                print("Total packets: {0}".format(count))
            else:
                print("No packets found")

        elif save_file is not None:
            with gzip.open(save_file, 'rb') as f:
                saved = pickle.load(f)
                self.filename = saved['filename']
                self.counter_time = saved['counter_time']
                self.total_livetime = saved['total_livetime']
                self.channel_livetime = saved['channel_livetime']
                self.channel_count = saved['channel_count']
                self.veto_count = saved['veto_count']
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
            save_file = self.filename + ".bgc.pgz"

        with gzip.open(save_file, 'wb') as f:
            pickle.dump({'filename' : self.filename,
                         'counter_time' : self.counter_time,
                         'total_livetime' : self.total_livetime,
                         'channel_livetime' : self.channel_livetime,
                         'channel_count' : self.channel_count,
                         'veto_count' : self.veto_count}, f, pickle.HIGHEST_PROTOCOL)

    @property
    def t(self):
        return self.counter_time

    @property
    def tl(self):
        return self.total_livetime

    @property
    def c(self):
        return self.channel_count

    @property
    def l(self):
        return self.channel_livetime

    @property
    def v(self):
        return self.veto_count
