"""
Module for analyzing card-cage information
"""
from __future__ import division, absolute_import, print_function

import os
from operator import attrgetter
import pickle
import gzip

import numpy as np
import pandas as pd

from ..telemetry import parser_generator
from ..util.time import oeb2utc

__all__ = ['CardCageInfo']


DIR = os.path.join(__file__, "..")


class CardCageInfo(object):
    """Class for analyzing card-cage information

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

            self.systime = [[], [], [], [], [], []]
            self.busy_fraction = [[], [], [], [], [], []]
            self.event_count = [[], [], [], [], [], []]

            count = 0

            print("Parsing {0}".format(telemetry_file))
            with open(telemetry_file, 'rb') as f:
                pg = parser_generator(f, filter_tmtype=0x08, verbose=True)
                for p in pg:
                    if p['systemid'] & 0xF0 != 0x80:
                        continue

                    count += 1
                    cc_number = p['systemid'] & 0x0F

                    self.systime[cc_number].append(p['systime'])
                    self.busy_fraction[cc_number].append(p['busy_time'] / p['elapsed_time'])
                    self.event_count[cc_number].append(p['event_count'])

            if count > 0:
                for i in range(6):
                    self.systime[i] = np.hstack(self.systime[i])
                    self.busy_fraction[i] = np.hstack(self.busy_fraction[i])
                    self.event_count[i] = np.hstack(self.event_count[i])

                print("Total packets: {0}".format(count))
            else:
                print("No packets found")

        elif save_file is not None:
            print("Restoring {0}".format(save_file))
            with gzip.open(save_file, 'rb') as f:
                saved = pickle.load(f)
                self.filename = saved['filename']
                self.systime = saved['systime']
                self.busy_fraction = saved['busy_fraction']
                self.event_count = saved['event_count']
        else:
            raise RuntimeError("Either a telemetry file or a save file must be specified")

    def __getitem__(self, key):
        if isinstance(key, int):
            if key >= 0 and key < 6:
                return pd.DataFrame({'busy_fraction' : self.busy_fraction[key],
                                     'event_count' : self.event_count[key]},
                                    index=oeb2utc(self.systime[key]))
            else:
                raise IndexError("Only integers from 0 to 5 are valid")
        if isinstance(key, str):
            out_dict = {}
            for i in range(6):
                out_dict['CC' + str(i)] = pd.Series(attrgetter(key)(self)[i], index=oeb2utc(self.systime[i]))
            return pd.DataFrame(out_dict)
        else:
            raise TypeError("Unsupported type")

    def save(self, save_file=None):
        """Save the parsed data for future reloading.
        The data is stored in gzip-compressed binary pickle format.

        Parameters
        ----------
        save_file : str
            The name of the save file to create.  If none is provided, the default is the name of
            the telemetry file with the extension ".ccinfo.pgz" appended.

        """
        if save_file is None:
            save_file = self.filename + ".ccinfo.pgz"

        with gzip.open(save_file, 'wb') as f:
            pickle.dump({'filename' : self.filename,
                         'systime' : self.systime,
                         'busy_fraction' : self.busy_fraction,
                         'event_count' : self.event_count}, f, pickle.HIGHEST_PROTOCOL)
