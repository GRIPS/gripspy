"""
Module for analyzing card-cage information
"""
from __future__ import division, absolute_import, print_function

import os
try:
    import cPickle as pickle
except ImportError:
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
    telemetry_file : str (or list)
        The name of the telemetry file to analyze.  If None is specified, a save file must be specified.
    save_file : str
        The name of a save file from a telemetry file that was previously parsed.

    Notes
    -----
    This implementation extracts all of the information, but the API is subject to change.
    """
    attributes_simple = ['systime', 'elapsed_time', 'busy_time', 'busy_count',
                              'veto_mult_trig_hard', 'veto_mult_trig_soft',
                              'busy_time_interface', 'busy_count_interface',
                              'busy_time_system', 'busy_count_system',
                              'guard_ring_lv_time', 'guard_ring_lv_count',
                              'guard_ring_hv_time', 'guard_ring_hv_count',
                              'shield_time', 'shield_count', 'reboot_count', 'event_count',
                              'veto_shield_soft', 'veto_guard_ring_lv_soft', 'veto_guard_ring_hv_soft',
                              'veto_shield_hard', 'veto_guard_ring_lv_hard', 'veto_guard_ring_hv_hard',
                              'dropped_event_count']
    attributes_all = attributes_simple + ['busy_fraction']

    def __init__(self, telemetry_file=None, save_file=None):
        if telemetry_file is not None:
            if isinstance(telemetry_file, list):
                self.filename = telemetry_file[0]
            else:
                self.filename = telemetry_file

            for attr in self.attributes_simple:
                setattr(self, attr, [[], [], [], [], [], []])
            self.busy_fraction = [None]*6

            count = 0

            print("Parsing {0}".format(self.filename))
            with open(self.filename, 'rb') as f:
                pg = parser_generator(f, filter_tmtype=0x08, verbose=True)
                for p in pg:
                    if p['systemid'] & 0xF0 != 0x80:
                        continue

                    count += 1
                    cc_number = p['systemid'] & 0x0F

                    for attr in self.attributes_simple:
                        getattr(self, attr)[cc_number].append(p[attr])

            if count > 0:
                for i in range(6):
                    for attr in self.attributes_simple:
                        getattr(self, attr)[i] = np.hstack(getattr(self, attr)[i])
                    self.busy_fraction[i] = self.busy_time[i] / self.elapsed_time[i]

                print("Total packets: {0}".format(count))
            else:
                print("No packets found")

            if isinstance(telemetry_file, list):
                for entry in telemetry_file[1:]:
                    self.append(CardCageInfo(entry))

        elif save_file is not None:
            if isinstance(save_file, list):
                to_open = save_file[0]
            else:
                to_open = save_file

            print("Restoring {0}".format(to_open))
            with gzip.open(to_open, 'rb') as f:
                try:
                    saved = pickle.load(f, encoding='latin1')
                except TypeError:
                    saved = pickle.load(f)
                self.filename = saved['filename']
                for attr in self.attributes_all:
                    setattr(self, attr, saved[attr])

            if isinstance(save_file, list):
                for entry in save_file[1:]:
                    self.append(CardCageInfo(save_file=entry))
        else:
            raise RuntimeError("Either a telemetry file or a save file must be specified")

    def __getitem__(self, key):
        if isinstance(key, int):
            if key >= 0 and key < 6:
                out_dict = {}
                for attr in self.attributes_all:
                    if attr == 'systime':
                        continue
                    out_dict[attr] = getattr(self, attr)[key]
                return pd.DataFrame(out_dict,
                                    index=oeb2utc(self.systime[key]))
            else:
                raise IndexError("Only integers from 0 to 5 are valid")
        if isinstance(key, str):
            out_dict = {}
            for i in range(6):
                out_dict['CC' + str(i)] = pd.Series(getattr(self, key)[i], index=oeb2utc(self.systime[i]))
            return pd.DataFrame(out_dict)
        else:
            raise TypeError("Unsupported type")

    def append(self, other):
        """Append the information in another CardCageInfo instance"""
        self.filename = (self.filename if type(self.filename) == list else [self.filename]) +\
                        (other.filename if type(other.filename) == list else [other.filename])
        for i in range(6):
            for attr in self.attributes_all:
                getattr(self, attr)[i] = np.hstack([getattr(self, attr)[i], getattr(other, attr)[i]])

    def save(self, save_file=None, use_current_directory=False):
        """Save the parsed data for future reloading.
        The data is stored in gzip-compressed binary pickle format.

        Parameters
        ----------
        save_file : str
            The name of the save file to create.  If none is provided, the default is the name of
            the telemetry file with the extension ".ccinfo.pgz" appended if a single telemetry file
            is the source.
        use_current_directory : bool
            If True, remove any directory specification from `save_file`

        Notes
        -----
        Save files may not be compatible between Python 2 and 3
        """
        if save_file is None:
            if type(self.filename) == str:
                to_save = self.filename + ".ccinfo.pgz"
            else:
                raise RuntimeError("The name for the save file needs to be explicitly specified here")
        else:
            to_save = save_file

        with gzip.open(to_save, 'wb') as f:
            print("Saving {0}".format(to_save))
            out_dict = {'filename' : self.filename}
            for attr in self.attributes_all:
                out_dict[attr] = getattr(self, attr)
            pickle.dump(out_dict, f, 2)
