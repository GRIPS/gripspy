"""
Module for analyzing data from the germanium detectors
"""
from __future__ import division, absolute_import, print_function

import os
from io import open
try:
    import cPickle as pickle
except ImportError:
    import pickle
import gzip
from copy import deepcopy

import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt

from ..telemetry import parser_generator
from ..util.time import oeb2utc


__all__ = ['GeData']


DIR = os.path.dirname(__file__)


stripmap = np.array([np.loadtxt(os.path.join(DIR, "asicmap", "asicmap{0}.txt".format(asic)), dtype=np.uint16)[:, 1]
                     for asic in range(8)]).flatten()


class GeData(object):
    """Class for analyzing event data from a germanium detector

    Parameters
    ----------
    detector : int
        The detector number (0 to 5, usually)
    telemetry_file : str (or list)
        The name of the telemetry file to analyze.  If None is specified, a save file must be specified.
    save_file : str (or list)
        The name of a save file from a telemetry file that was previously parsed.
    max_triggers : int
        If not None, cull any events with more than this number of triggers on either detector side.

    Notes
    -----
    `event_time` is stored in 10-ns steps
    """
    def __init__(self, detector, telemetry_file=None, save_file=None, max_triggers=None):
        self.detector = detector

        if telemetry_file is not None:
            if isinstance(telemetry_file, list):
                self.filename = telemetry_file[0]
            else:
                self.filename = telemetry_file

            print("Parsing {0}".format(self.filename))
            result = process(self.filename, detector, os.path.join(DIR, "cms", "cc{0}_1000".format(detector)))
            self.adc = result[0]
            self.cms = result[1]
            self.delta_time = result[2]
            self.event_time = result[3]
            self.glitch = result[4]
            self.trigger = result[5]
            self.veto = result[6]

            if max_triggers is not None:
                self._cull_excessive_triggers(max_triggers)

            if isinstance(telemetry_file, list):
                for entry in telemetry_file[1:]:
                    self.append(GeData(self.detector, entry, max_triggers=max_triggers), defer_processing=True)

            self._process_single_triggers()

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
                self.adc = saved['adc']
                self.cms = saved['cms']
                self.delta_time = saved['delta_time']
                self.event_time = saved['event_time']
                self.glitch = saved['glitch']
                self.trigger = saved['trigger']
                self.veto = saved['veto']

            if max_triggers is not None:
                self._cull_excessive_triggers(max_triggers)

            if isinstance(save_file, list):
                for entry in save_file[1:]:
                    self.append(GeData(self.detector, save_file=entry, max_triggers=max_triggers), defer_processing=True)

            self._process_single_triggers()

        else:
            raise RuntimeError("Either a telemetry file or a save file must be specified")

    def __add__(self, other):
        out = deepcopy(self)
        out.append(other)
        return out

    def _cull_excessive_triggers(self, max_triggers):
        """Only keep events that do not exceed a maximum number of triggers on each side."""
        keep = np.flatnonzero(np.logical_and(self.trigger[:, 0:256].sum(1) <= max_triggers,
                                             self.trigger[:, 256:512].sum(1) <= max_triggers).A1)
        self.adc = self.adc[keep, :]
        self.cms = self.cms[keep, :]
        self.delta_time = self.delta_time[keep, :]
        self.event_time = self.event_time[keep]
        self.glitch = self.glitch[keep, :]
        self.trigger = self.trigger[keep, :]
        self.veto = self.veto[keep, :]

    def _process_single_triggers(self):
        self.single_triggers = np.flatnonzero(np.logical_and(self.trigger[:, 0:256].sum(1) == 1,
                                                             self.trigger[:, 256:512].sum(1) == 1).A1)
        if len(self.single_triggers) == 0:
            self.single_triggers = np.flatnonzero(np.logical_or(self.trigger[:, 0:256].sum(1) == 1,
                                                                self.trigger[:, 256:512].sum(1) == 1).A1)

        self.single_triggers_lv = self.trigger[self.single_triggers, 0:256].nonzero()[1]
        self.single_triggers_hv = self.trigger[self.single_triggers, 256:512].nonzero()[1] + 256

    def append(self, other, defer_processing=False):
        """Append the information in another GeData instance"""
        self.filename = (self.filename if type(self.filename) == list else [self.filename]) +\
                        (other.filename if type(other.filename) == list else [other.filename])
        self.adc = sps.vstack([self.adc, other.adc])
        self.cms = sps.vstack([self.cms, other.cms])
        self.delta_time = sps.vstack([self.delta_time, other.delta_time])
        self.event_time = np.hstack([self.event_time, other.event_time])
        self.glitch = sps.vstack([self.glitch, other.glitch])
        self.trigger = sps.vstack([self.trigger, other.trigger])
        self.veto = sps.vstack([self.veto, other.veto])

        if not defer_processing:
            self._process_single_triggers()

    def save(self, save_file=None, use_current_directory=False):
        """Save the parsed data for future reloading.
        The data is stored in gzip-compressed binary pickle format.

        Parameters
        ----------
        save_file : str
            The name of the save file to create.  If none is provided, the default is the name of
            the telemetry file with the extension ".ge?.pgz" appended if a single telemetry file
            is the source.
        use_current_directory : bool
            If True, remove any directory specification from `save_file`

        Notes
        -----
        Save files may not be compatible between Python 2 and 3
        """
        if save_file is None:
            if type(self.filename) == str:
                to_save = self.filename + ".ge{0}.pgz".format(self.detector)
            else:
                raise RuntimeError("The name for the save file needs to be explicitly specified here")
        else:
            to_save = save_file

        if use_current_directory:
            to_save = os.path.basename(to_save)

        with gzip.open(to_save, 'wb') as f:
            print("Saving {0}".format(to_save))
            pickle.dump({'filename' : self.filename,
                         'adc' : self.adc,
                         'cms' : self.cms,
                         'delta_time' : self.delta_time,
                         'event_time' : self.event_time,
                         'glitch' : self.glitch,
                         'trigger' : self.trigger,
                         'veto' : self.veto}, f, 2)

    @property
    def a(self):
        """Shorthand for the `adc` attribute"""
        return self.adc

    @property
    def c(self):
        """Shorthand for the `cms` attribute"""
        return self.cms

    @property
    def d(self):
        """Shorthand for the `delta_time` attribute"""
        return self.delta_time

    @property
    def e(self):
        """Shorthand for the `event_time` attribute"""
        return self.event_time

    @property
    def g(self):
        """Shorthand for the `glitch` attribute"""
        return self.glitch

    @property
    def t(self):
        """Shorthand for the `trigger` attribute"""
        return self.trigger

    @property
    def v(self):
        """Shorthand for the `veto` attribute"""
        return self.veto

    @property
    def vl(self):
        """Shorthand for the LV guard-ring vetoes"""
        return self.veto[:, 0]

    @property
    def vh(self):
        """Shorthand for the HV guard-ring vetoes"""
        return self.veto[:, 1]

    @property
    def vs(self):
        """Shorthand for the shield vetoes"""
        return self.veto[:, 2]

    @property
    def vm(self):
        """Shorthand for the multiple-trigger vetoes"""
        return self.veto[:, 3]

    @property
    def s(self):
        """Shorthand for the `single_triggers` attribute"""
        return self.single_triggers

    @property
    def s_lv(self):
        """Shorthand for the `single_triggers_lv` attribute"""
        return self.single_triggers_lv

    @property
    def s_hv(self):
        """Shorthand for the `single_triggers_lv` attribute"""
        return self.single_triggers_hv

    @property
    def hitmap(self):
        """The hitmap of single-trigger events"""
        lv = stripmap[self.s_lv]
        hv = stripmap[self.s_hv]
        good = np.logical_and(lv < 900, hv < 900)
        return sps.coo_matrix((np.ones(np.sum(good)), (hv[good], lv[good])), shape=(150, 150), dtype=int).toarray()

    def lightcurve(self, side, binning, time_step_in_seconds=1.,
                   energy_coeff=None, max_triggers=1):
        """Return a detector-side-integrated lightcurve, excluding events that exceed a specified
        maximum number of triggers on that side.  Note that `max_triggers` == is not quite the
        same as "single-trigger" events because no selection cut is imposed on the the opposite
        side.

        Parameters
        ----------
        side : 0 or 1
            0 for LV side, 1 for HV side
        binning : array-like
            The binning to use for the underlying data
        time_step_in_seconds : float
            The size of the time step in seconds
        energy_coeff: 2x512 `~numpy.ndarray`
            If not None, apply these linear coefficients (intercept, slope) to convert to energy in keV
        max_triggers : int
            Exclude events that have more than this number of triggers on the specified side
        """
        channels = slice(0, 256) if side == 0 else slice(256, 512)
        events = (self.t[:, channels].sum(1) <= max_triggers).A1

        x_values = self.e[events]

        y_values = self.c[events, channels].multiply(self.t[events, channels])
        if energy_coeff is not None:
            y_values = self.t[events, channels] * energy_coeff[0, channels] +\
                       y_values.asfptype() * energy_coeff[1, channels]
        else:
            y_values = y_values.sum(1).A1

        x_binning = np.arange(np.min(self.e),
                              np.max(x_values) + time_step_in_seconds * 1e8,
                              time_step_in_seconds * 1e8)

        time_axis = (x_binning[:-1] + x_binning[1:]) / 2
        h = np.histogram2d(y_values, x_values, [binning, x_binning])

        return oeb2utc(time_axis / 10), h[0].T

    def plot_depth(self, binning=np.arange(-595, 596, 10)):
        """Plot the depth-information plot
        
        Parameters
        ----------
        binning : array-like
            The binning to use for the underlying data
        """
        plt.hist((self.d[self.s, self.s_hv].A1 - self.d[self.s, self.s_lv].A1) * 10.,
                 bins=binning, histtype='step', label='CC{0}'.format(self.detector))
        plt.xlabel("Nanoseconds")
        plt.title("CC{0} HV time minus LV time (i.e., left is HV side)".format(self.detector))
        return plt.gca()

    def plot_hitmap(self, **imshow_kwargs):
        """Plot the hitmap of single-trigger events"""
        args = {'origin' : 'lower',
                'interpolation' : 'nearest',
                'cmap' : 'gray'}
        args.update(imshow_kwargs)
        plt.imshow(self.hitmap, **args)
        plt.title("CC{0}".format(self.detector))
        plt.xlabel("LV (ASICs 1/3 to 0/2)")
        plt.ylabel("HV (ASICs 4/6 to 5/7)")
        plt.colorbar()
        return plt.gca()

    def plot_multiple_trigger_veto(self, side):
        """Plot the distribution of multiple-trigger events and veto information

        Parameters
        ----------
        side : 0 or 1
            0 for LV side, 1 for HV side
        """
        side %= 2
        plt.hist(self.t[:, side*256:(side+1)*256].sum(1).A1, bins=np.arange(1, 151), histtype='step', label='All')
        plt.hist(self.t[:, side*256:(side+1)*256].sum(1).A1[~self.vm.todense().A1], bins=np.arange(1, 151), histtype='step', label='Not vetoed')
        plt.hist(np.logical_and(self.d[:, side*256:(side+1)*256].toarray() <= 126, self.t[:, side*256:(side+1)*256].toarray())[~self.vm.todense().A1, :].sum(1), bins=np.arange(1, 151), histtype='step', label='Not vetoed and delta <= 126')
        plt.xlabel("Number of channels")
        plt.title("CC{0} {1} side".format(self.detector, "LV" if side == 0 else "HV"))
        plt.legend()
        plt.semilogy()
        return plt.gca()

    def plot_spatial_spectrum(self, side, binning=np.arange(-128, 2048, 8),
                              energy_coeff=None, use_strip_number=False, **hist2d_kwargs):
        """Plot the subtracted single-trigger spectrum versus channel as a 2D image

        Parameters
        ----------
        side : 0 or 1
            0 for LV side, 1 for HV side
        binning : array-like
            The binning to use for the underlying data
        energy_coeff: 2x512 `~numpy.ndarray`
            If not None, apply these linear coefficients (intercept, slope) to convert to energy in keV
        use_strip_number: bool
            If True, use strip number as the horizontal axis instead of channel number
        """
        s_side = self.s_lv if side == 0 else self.s_hv
        x_values = stripmap[s_side] if use_strip_number else s_side
        y_values = self.c[self.s, s_side].A1
        if energy_coeff is not None:
            y_values = y_values.astype('float') * energy_coeff[1, s_side] + energy_coeff[0, s_side]

        x_binning = np.arange(1, 150) if use_strip_number else np.arange(side*256, (side+1)*256+1)
        args = {'bins' : [x_binning, binning],
                'cmap' : 'gray'}
        args.update(hist2d_kwargs)
        plt.hist2d(x_values, y_values, **args)

        if use_strip_number:
            plt.xlabel("Strip number (1 to 149)")
        else:
            plt.xlabel("Channel number " + ("(0 to 255)" if side == 0 else "(256 to 511)"))
        if energy_coeff is not None:
            plt.ylabel("Energy (keV)")
        else:
            plt.ylabel("Subtracted ADC value")
        plt.title("CC{0} {1} side".format(self.detector, "LV" if side == 0 else "HV"))

        return plt.gca()

    def plot_spectrum(self, asiccha, binning=np.arange(0, 3584, 8)):
        """Plot the raw spectrum for a specified channel

        Parameters
        ----------
        asiccha : tuple or int
            Either (ASIC#, channel#) if a tuple or ASIC# * 64 + channel# if an int
        binning : array-like
            The binning to use for the underlying data
        """
        if type(asiccha) == tuple:
            index = asiccha[0] * 64 + asiccha[1]
        else:
            index = asiccha

        try:
            val, _, _ = plt.hist(self.a[:, index][self.t[:, index].nonzero()].A1, bins=binning, histtype='step', color='r', label='All triggers')
            plt.ylim(0, np.max(val))
        except AttributeError: # The channel never triggered
            pass
        try:
            plt.hist(self.a[self.s, index][self.t[self.s, index].nonzero()].A1, bins=binning, histtype='step', color='k', label='Single triggers')
        except AttributeError: # The channel never triggered as a single-trigger pair
            pass
        plt.hist(self.a[:, index].toarray()[~self.t[:, index].toarray()], bins=binning, histtype='step', color='b', label='Untriggered')
        plt.legend()
        plt.title("Raw ADC spectra for CC{0}/A{1}-{2}".format(self.detector, *divmod(index, 64)))
        return plt.gca()

    def plot_spectrogram(self, side, time_step_in_seconds=1., binning=np.arange(-128, 2048, 8),
                         energy_coeff=None, max_triggers=1, **hist2d_kwargs):
        """Plot a detector-side-integrated spectrogram, excluding events that exceed a specified
        maximum number of triggers on that side.  Note that `max_triggers` == is not quite the
        same as "single-trigger" events because no selection cut is imposed on the the opposite
        side.

        Parameters
        ----------
        side : 0 or 1
            0 for LV side, 1 for HV side
        time_step_in_seconds : float
            The size of the time step in seconds
        binning : array-like
            The binning to use for the underlying data
        energy_coeff: 2x512 `~numpy.ndarray`
            If not None, apply these linear coefficients (intercept, slope) to convert to energy in keV
        max_triggers : int
            Exclude events that have more than this number of triggers on the specified side
        """
        channels = slice(0, 256) if side == 0 else slice(256, 512)
        events = (self.t[:, channels].sum(1) <= max_triggers).A1

        x_values = (self.e[events] - np.min(self.e)) / 1e8

        y_values = self.c[events, channels].multiply(self.t[events, channels])
        if energy_coeff is not None:
            y_values = self.t[events, channels] * energy_coeff[0, channels] +\
                       y_values.asfptype() * energy_coeff[1, channels]
        else:
            y_values = y_values.sum(1).A1

        x_binning = np.arange(0, np.max(x_values) + time_step_in_seconds, time_step_in_seconds)
        args = {'bins' : [x_binning, binning],
                'cmap' : 'gray'}
        args.update(hist2d_kwargs)
        plt.hist2d(x_values, y_values, **args)

        plt.xlabel("Time (s)")
        if energy_coeff is not None:
            plt.ylabel("Energy (keV)")
        else:
            plt.ylabel("Subtracted ADC value")
        plt.title("CC{0} {1} side".format(self.detector, "LV" if side == 0 else "HV"))

        return plt.gca()

    def plot_subtracted_spectrum(self, asiccha, binning=np.arange(-128, 2048, 8)):
        """Plot the common-mode-subtracted spectrum for a specified channel

        Parameters
        ----------
        asiccha : tuple or int
            Either (ASIC#, channel#) if a tuple or ASIC# * 64 + channel# if an int
        binning : array-like
            The binning to use for the underlying data
        """
        if type(asiccha) == tuple:
            index = asiccha[0] * 64 + asiccha[1]
        else:
            index = asiccha

        try:
            plt.hist(self.c[:, index][self.t[:, index].nonzero()].A1, bins=binning, histtype='step', color='r', label='All triggers')
        except AttributeError: # The channel never triggered
            pass
        try:
            plt.hist(self.c[self.s, index][self.t[self.s, index].nonzero()].A1, bins=binning, histtype='step', color='k', label='Single triggers')
        except AttributeError: # The channel never triggered as a single-trigger pair
            pass
        plt.legend()
        plt.title("Common-mode-subtracted ADC spectra for CC{0}/A{1}-{2}".format(self.detector, *divmod(index, 64)))
        return plt.gca()


def accumulate(f, cc_number, max_events=10000, verbose=False):
    """Return four arrays from a telemetry-file object for the specified card cage:
    ADC values, delta (time difference) values, glitch flags, and trigger flags.
    The procedure will stop if the maximum number of events is reached.

    To read an unknown number of events, use `bin_to_sparse_arrays()`.

    Parameters
    ----------
    f : file object

    cc_number : int
        The number of the desired card cage (0-5)

    max_events : int
        The accumulation will stop after this many events (default is 10000)

    verbose : boolean
        If True, output a percentage progress

    Returns
    -------
    adc : `~numpy.ndarray`
        Nx512 int16 array of the ADC values, where N <= max_events

    delta_time : `~numpy.ndarray`
        Nx512 int8 array of the delta (time difference) values, where N <= max_events

    event_time : `~numpy.ndarray`
        N int64 array of the event times in 10-ns steps, where N <= max_events

    glitch : `~numpy.ndarray`
        Nx512 bool array of the glitch flags, where N <= max_events

    trigger : `~numpy.ndarray`
        Nx512 bool array of the trigger flags, where N <= max_events

    veto : `~numpy.ndarray`
        Nx4 bool array of the veto flags, where N <= max_events
    """
    event_gen = parser_generator(f, filter_systemid=0x80 + cc_number, filter_tmtype=0xF3, verbose=verbose)
    adc = np.zeros((max_events, 8 * 64), dtype=np.int16)
    delta_time = np.full((max_events, 8 * 64), -128, dtype=np.int8)
    event_time = np.zeros(max_events, dtype=np.int64)
    glitch = np.zeros((max_events, 8 * 64), dtype=np.bool)
    trigger = np.zeros((max_events, 8 * 64), dtype=np.bool)
    veto = np.zeros((max_events, 4), dtype=np.bool)
    counter = 0

    for event in event_gen:
        adc[counter, event['asiccha']] = event['adc']
        delta_time[counter, event['asiccha']] = event['delta_time']
        event_time[counter] = event['event_time']
        glitch[counter, event['asiccha']] = event['has_glitch']
        trigger[counter, event['asiccha']] = event['has_trigger']
        veto[counter, :] = [event['veto_lvgr'], event['veto_hvgr'], event['veto_shield'], event['veto_multiple']]

        counter += 1
        if counter == max_events:
            break

    if counter == 0:
        raise ValueError("No events found for card cage {0}!".format(cc_number))

    if verbose:
        print("Accumulated {0} events".format(counter))

    return (adc[:counter, :],
            delta_time[:counter, :],
            event_time[:counter],
            glitch[:counter, :],
            trigger[:counter, :],
            veto[:counter, :])


def bin_to_sparse_arrays(one_or_more_files, cc_number, max_events=None, verbose=True,
    pedestals=None, cm_matrix=None, scale=None):
    """Returns six (sparse) arrays from one or more binary event files for the specified card cage.
    The column index (0 to 511) is ASIC number times 64 plus channel number.

    Parameters
    ----------
    one_or_more_files : string or list of strings
        One or more binary event files to convert to arrays

    cc_number : int
        The number of the desired card cage (0-5)

    max_events : int
        If specified, the accumulation will stop after this many events

    verbose : boolean
        If True, output a percentage progress

    pedestals : `~numpy.ndarray`
        8x64 int16 array of the pedestal locations

    cm_matrix : `~numpy.ndarray`
        8x64x64 boolean array of the common-mode matrices, pedestals must also be provided

    scale : `~numpy.ndarray`
        8x64x64 float array of the channel-channel ratios of their noise amplitudes

    Returns
    -------
    adc : `~scipy.sparse.csr_matrix`
        Nx512 int16 array of the ADC values

    cms : `~scipy.sparse.csr_matrix`
        Nx512 int16 array of the common-mode-subtracted ADC values (or None, if metadata is not provided)

    delta_time : `~scipy.sparse.csr_matrix`
        Nx512 int8 array of the delta (time difference) values

    event_time : `~numpy.ndarray`
        N int64 array of the event times in 10-ns steps

    glitch : `~scipy.sparse.csr_matrix`
        Nx512 bool array of the glitch flags

    trigger : `~scipy.sparse.csr_matrix`
        Nx512 bool array of the trigger flags

    veto : `~scipy.sparse.csr_matrix`
        Nx4 bool array of the veto flags, where N <= max_events
    """
    filelist = one_or_more_files if type(one_or_more_files) == list else [one_or_more_files]

    # Read the file in "chunks" of 10000 events for better memory management of arrays
    chunk_size = 10000

    adc = []
    cms = []
    delta_time = []
    event_time = []
    glitch = []
    trigger = []
    veto = []
    
    total = 0

    for entry in filelist:
        with open(entry, "rb") as f:
            #print("Converting file {0} to sparse arrays".format(entry))
            while True:
                # Read in the next chunk
                chunk = accumulate(f, cc_number, max_events=chunk_size, verbose=verbose)
                total += len(chunk[2])

                # Convert to sparse matrices, COO format to use row/col indices
                adc_chunk = sps.coo_matrix(chunk[0])
                trigger_chunk = sps.coo_matrix(chunk[4])
                # Care needs to be taken with delta_time because 0 is a valid value
                delta_time_chunk = sps.coo_matrix((chunk[1][trigger_chunk.row, trigger_chunk.col], (trigger_chunk.row, trigger_chunk.col)), shape=trigger_chunk.shape, dtype=np.int8)

                # Apply common-mode subtraction if the metadata is provided
                if pedestals is not None and cm_matrix is not None and scale is not None:
                    pedestal_subtracted = chunk[0] - np.expand_dims(pedestals.reshape(512), axis=0)
                    common_mode_subtracted = np.zeros_like(pedestal_subtracted)
                    # Step through every channel that has a common-mode group
                    for asic in range(8):
                        for channel in np.flatnonzero(np.sum(cm_matrix[asic, :, :], axis=1)):
                            # This crazy line subtracts the median of the scaled ADC values from the common-mode group for the entire chunk at once
                            common_mode_subtracted[:, asic * 64 + channel] = pedestal_subtracted[:, asic * 64 + channel] - np.median((pedestal_subtracted[:, asic * 64:(asic + 1) * 64] / np.expand_dims(scale[asic, channel, :], axis=0))[:, cm_matrix[asic, channel, :]], axis=1)
                    # Create the CMS sparse matrix using the same row and column indices
                    cms_chunk = sps.coo_matrix((common_mode_subtracted[adc_chunk.row, adc_chunk.col], (adc_chunk.row, adc_chunk.col)), shape=adc_chunk.shape, dtype=np.int16)
                    cms.append(cms_chunk)

                adc.append(adc_chunk)
                delta_time.append(delta_time_chunk)
                event_time.append(chunk[2])
                glitch.append(sps.csr_matrix(chunk[3]))
                trigger.append(trigger_chunk)
                veto.append(sps.csr_matrix(chunk[5]))

                # Break out if enough chunks have been read
                if max_events is not None and total >= max_events:
                    break

                # Break out if the end of file has been reached
                current_location = f.tell()
                f.seek(0, 2)
                end_of_file = f.tell()
                if current_location == end_of_file:
                    break
                f.seek(current_location)

    # Merge the arrays from chunks and trim out unused depth
    adc = sps.vstack(adc, 'csr')
    delta_time = sps.vstack(delta_time, 'csr')
    event_time = np.hstack(event_time)
    glitch = sps.vstack(glitch, 'csr')
    trigger = sps.vstack(trigger, 'csr')
    veto = sps.vstack(veto, 'csr')
    if len(cms) > 0:
        cms_out = sps.vstack(cms, 'csr')
    else:
        cms_out = None
    if max_events is not None:
        adc = adc[:max_events, :]
        delta_time = delta_time[:max_events, :]
        event_time = event_time[:max_events]
        glitch = glitch[:max_events, :]
        trigger = trigger[:max_events, :]
        veto = veto[:max_events, :]
        if cms_out is not None: cms_out[:max_events, :]

    print("Total events: {0}".format(len(event_time)))

    return (adc, cms_out, delta_time, event_time, glitch, trigger, veto)


def process(one_or_more_files, cc_number, identifier):
    """This will perform common-mode subtraction using the CMS.NPZ file
    generated in step one and output six (sparse) arrays (see below).

    Parameters
    ----------
    one_or_more_files : string or list of strings
        One or more files to parse

    cc_number : int
        The number of the desired card cage (0-5)
    
    identifier : string
        The identifier used in the first step to create the appropriate .cms.npz file

    cms : boolean
        If True, perform common-mode subtraction, otherwise perform only pedestal subtraction

    Returns
    -------
    adc : `~scipy.sparse.csr_matrix`
        Nx512 int16 array of the ADC values

    cms : `~scipy.sparse.csr_matrix`
        Nx512 int16 array of the common-mode-subtracted ADC values

    delta_time : `~scipy.sparse.csr_matrix`
        Nx512 int8 array of the delta (time difference) values

    event_time : `~numpy.ndarray`
        N int64 array of the event times in 10-ns steps

    glitch : `~scipy.sparse.csr_matrix`
        Nx512 bool array of the glitch flags

    trigger : `~scipy.sparse.csr_matrix`
        Nx512 bool array of the trigger flags

    veto : `~scipy.sparse.csr_matrix`
        Nx4 bool array of the veto flags, where N <= max_events
    """
    filelist = one_or_more_files if type(one_or_more_files) == list else [one_or_more_files]

    with np.load("{0}.cms.npz".format(identifier)) as cms_npz:
        pedestals = cms_npz["pedestals"]
        cm_matrix = cms_npz["cm_matrix"]
        scale = cms_npz["scale"]

        return bin_to_sparse_arrays(filelist, cc_number, pedestals=pedestals, cm_matrix=cm_matrix, scale=scale)
