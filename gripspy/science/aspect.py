"""
Module for processing aspect data
"""
from __future__ import division, absolute_import, print_function

from sys import stdout
from operator import attrgetter, itemgetter
import warnings
import inspect

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import pandas as pd
from scipy.optimize import curve_fit
from skimage.feature import match_template, peak_local_max
from skimage.transform import warp
from astropy.io import fits

from ..util.time import oeb2utc
from ..util.fitting import parapeak


__all__ = ['PYFrame', 'PYSequence', 'RFrame', 'RSequence']


NUM_ROWS = 960
NUM_COLS = 1280


def _kernel(radius):
    """The empirically chosen kernel to represent both the Sun and the fiducials."""
    axis = np.arange(-(radius * 2), radius * 2 + 1)
    x, y = np.meshgrid(axis, axis)
    template = 1 - np.cosh(np.sqrt(x**2 + y**2) * 5. / radius) / 200.
    template[template < 0] = 0
    return template


template_sun = _kernel(50)
template_fiducial = 1 - _kernel(4)


class Frame(object):
    """Base class for a camera frame

    Parameters
    ----------
    filename : str
        The name of the FITS file of the camera frame
    uid : int
        The UID of the camera to check for consistency with the camera frame.
        The default of None means that consistency is not checked.
    decimation_range : tuple of int
        The allowable range of decimation levels.
    """
    def __init__(self, filename, uid=None, decimation_range=(2, 10)):
        with fits.open(filename) as f:
            if uid != None and f[0].header['CAMERA'] != uid:
                raise RuntimeError
            self.filename = filename
            self.header = f[0].header
            self.data = f[1].data.astype(np.uint8)
            self.trigger_time = f[0].header['GT_TRIG']
        self.decimation = self._detect_decimation(decimation_range)

    def _detect_decimation(self, decimation_range):
        """Detects the decimation level of the camera frame.

        Parameters
        ----------
        decimation_range : tuple of int
            The inclusive range of decimation levels to check against the camera frame

        Returns
        -------
        decimation_level : int
            For a decimation_level of N, N-1 rows out of every N rows is empty
        decimation_index : int
            The index of the non-empty row for every decimation block
            (i.e. strictly less than decimation_level)
        """
        dead1 = (self.data == 0).sum(1) == NUM_COLS
        for step in range(decimation_range[0], decimation_range[1] + 1):
            sub_rows = NUM_ROWS // step
            dead2 = dead1[:sub_rows * step].reshape(sub_rows, step).max(0)
            idx = np.flatnonzero(~dead2)
            if len(idx) == 1:
                return (step, idx[0])
        return (1, 0)

    def plot_image(self, **imshow_kwargs):
        """Plots the camera frame, defaulting to no interpolation."""
        args = {'cmap'   : 'gray',
                'extent' : [-0.5, NUM_COLS - 0.5, NUM_ROWS - 0.5, -0.5],
                'interpolation' : 'nearest'}
        args.update(imshow_kwargs)
        return plt.imshow(self.image, **args)

    @property
    def image(self):
        if self.decimation == (1, 0):
            return self.data
        else:
            transformation = np.matrix([[1, 0, 0],
                                        [0, 1. / self.decimation[0], -self.decimation[1] / self.decimation[0]],
                                        [0, 0, 1]])
            return warp(self.data[self.decimation[1]::self.decimation[0], :], transformation,
                        output_shape=(NUM_ROWS, NUM_COLS), mode='edge', preserve_range=True)


class PYFrame(Frame):
    """Class for a pitch-yaw image

    Parameters
    ----------
    filename : str
        The name of the FITS file of the camera frame
    uid : int
        Defaults to the UID of the pitch-yaw camera, but can be set to a different camera's UID or to None
    """
    def __init__(self, filename, uid=158434):
        try:
            Frame.__init__(self, filename, uid=uid)
        except RuntimeError:
            raise RuntimeError("This file does not appear to be a valid frame from the pitch-yaw camera")

        self.peaks, self.fiducials = self._process()

    def _process(self):
        """Finds the Suns and the fiducials."""
        # Perform a coarse search for Suns
        coarse_image = self.image[::10, ::10]
        coarse_match = match_template(coarse_image, template_sun[::10, ::10], pad_input=True)
        coarse_peaks = peak_local_max(coarse_match, threshold_abs=0.9, num_peaks=3)

        fine_peaks = []
        strength = []
        fiducials = []

        for coarse_peak in coarse_peaks:
            # For each coarse detection, do a detection at the full resolution
            if coarse_peak[0] < 11 or coarse_peak[0] > 84 or coarse_peak[1] < 11 or coarse_peak[1] > 116:
                break
            sub_image = self.image[coarse_peak[0] * 10 - 110:coarse_peak[0] * 10 + 111,
                                   coarse_peak[1] * 10 - 110:coarse_peak[1] * 10 + 111]
            match = match_template(sub_image, template_sun, pad_input=True)
            peak = peak_local_max(match, threshold_abs=0.9, num_peaks=1)
            if len(peak) > 0:
                peak = peak[0]
                peak_r, peak_c = parapeak(match[peak[0] - 1:peak[0] + 2, peak[1] - 1:peak[1] + 2])
                peak += coarse_peak * 10 - 110

                fine_peaks.append((peak[0] + peak_r, peak[1] + peak_c))

                #FIXME: need a more robust estimate of the strength of each peak
                strength.append(self.image[peak[0], peak[1]])

                # Find fiducials near the center of the Sun
                match = match_template(self.image[peak[0]-60:peak[0]+61, peak[1]-60:peak[1]+61],
                                       template_fiducial, pad_input=True)
                fids = peak_local_max(match, threshold_abs=0.8)
                for fid in fids:
                    fid_r, fid_c = parapeak(match[fid[0] - 1:fid[0] + 2, fid[1] - 1:fid[1] + 2])
                    fid += peak - 60

                    fiducials.append((fid[0] + fid_r, fid[1] + fid_c))

        # Sort the peaks in order of decreasing strength
        fine_peaks = [peak for (strength, peak) in sorted(zip(strength, fine_peaks), reverse=True)]

        return fine_peaks, fiducials

    def plot_image(self, **imshow_kwargs):
        """Plots the pitch-yaw image, including Sun/fiducial detections."""
        if self.peaks:
            # Draw an X at the center of each Sun
            arr_peaks = np.array(self.peaks)
            plt.plot(arr_peaks[:, 1], arr_peaks[:, 0], 'rx')

            # Draw circles around each Sun
            radius = 60
            angle = 2 * np.pi * np.arange(0, 101, 0.01)
            for peak in self.peaks:
                plt.plot(peak[1] + radius * np.cos(angle), peak[0] + radius * np.sin(angle), 'r')

        if self.fiducials:
            # Draw a cross at each fiducial
            arr_fiducials = np.array(self.fiducials)
            plt.plot(arr_fiducials[:, 1], arr_fiducials[:, 0], 'b+')

        return Frame.plot_image(self, **imshow_kwargs)


class RFrame(Frame):
    """Class for a roll image

    Parameters
    ----------
    filename : str
        The name of the FITS file of roll image
    uid : int
        Defaults to the UID of the roll camera, but can be set to a different camera's UID or to None
    """
    def __init__(self, filename, uid=142974):
        try:
            Frame.__init__(self, filename, uid=uid, decimation_range=(10, 10))
        except RuntimeError:
            raise RuntimeError("This file does not appear to be a valid frame from the roll camera")

        self.midpoint = self._process()

    def _process(self, plot=False):
        """Fit the horizons"""
        # For now, fit the whole thing
        data = np.mean(self.data[self.decimation[1]::self.decimation[0], :], 0)
        left = np.min(np.flatnonzero(data[0:640] < np.mean(data[0:640])))
        right = np.max(np.flatnonzero(data[640:] < np.mean(data[640:]))) + 640

        subx = np.arange(left, right + 1)
        guess = (639.5, 0.02, 0.04, 10)
        try:
            popt, pcov = curve_fit(RFrame.atmosphere_sym, subx, data[left:right+1], guess)
        except RuntimeError:
            warnings.warn_explicit("Could not automatically fit the horizon",
                                   RuntimeWarning, __file__, inspect.currentframe().f_lineno)
            return None

        if plot:
            print(popt)

            plt.plot(data)
            plt.plot(subx, RFrame.atmosphere_sym(subx, *popt))

        return popt[0]

    def plot_image(self, **imshow_kwargs):
        """Plots the roll image, including annotations."""
        return Frame.plot_image(self, **imshow_kwargs)

    @staticmethod
    def atmosphere_sym(x, midpoint, scale, amp, dc):
        """Exponential atmospheric model with symmetry (i.e., hyperbolic cosine)"""
        return amp * np.cosh(scale * (x - midpoint)) + dc

    @staticmethod
    def atmosphere_asym(x, midpoint, scale, amp1, amp2, dc):
        """Exponential atmospheric model with asymmetry in normalization"""
        return amp1 * np.exp(-scale * (x - midpoint)) + amp2 * np.exp(scale * (x - midpoint)) + dc


class FrameSequence(list):
    """Base class for a sequence of camera frames"""
    def __init__(self, list_of_files, frame_class=Frame):
        for entry in list_of_files:
            self.append(frame_class(entry))
            stdout.write(".")

        stdout.write("\n")

    def animate(self, fps=4, fig=None, **imshow_kwargs):
        """Return a matplotlib animation of the frame sequence.  Extra keyword arguments are passed on to
        `matplotlib.pyplot.imshow`.

        Parameters
        ----------
        fps : int
            Number of frames per second for interactive display

        fig : `matplotlib.figure.Figure`
            If None, the current figure is used

        Returns
        -------
        `matplotlib.animation.FuncAnimation`

        Examples
        --------
        To save a matplotlib animation as an animated GIF:

        >>> anim = framesequence.animate()
        >>> anim.save('animation.gif', writer='imagemagick', fps=4)

        On Windows, you may need to configure matplotlib to find `convert.exe`, e.g.:

        >>> import matplotlib as mpl
        >>> mpl.rcParams['animation.convert_path'] = 'C:\\Program Files\\ImageMagick-6.9.2-Q16\\convert.exe'

        """
        if fig is None:
            fig = plt.gcf()

        args = {'cmap'   : 'gray',
                'extent' : [-0.5, NUM_COLS - 0.5, NUM_ROWS - 0.5, -0.5],
                'clim' : (0, 256),
                'interpolation' : 'nearest'}
        args.update(imshow_kwargs)
        im = plt.imshow(np.zeros_like(self[0].image), **args)

        iterator = self.__iter__()

        # Although this function does nothing, otherwise matplotlib "burns" the first frame for initialization
        def init():
            pass

        # Update with the next frame's data
        def update_im(frame, im):
            im.set_data(frame.image)
            plt.gca().set_title(frame.filename)
            return im,

        ani = animation.FuncAnimation(fig, update_im, iterator, fargs=(im,), init_func=init,
                                      repeat=False, interval=1000 / fps)

        return ani

    def compact(self):
        """Crude compaction of a decimated sequence"""
        for i in range(len(self) - 1, 0, -1):
            if (self[i].decimation[0] == self[i-1].decimation[0]) and\
               (self[i].decimation[1] > self[i-1].decimation[1]):
                self[i-1].data += self[i].data
                self.pop(i)
            else:
                self[i].decimation = (1, 0)
        self[0].decimation = (1, 0)


class PYSequence(FrameSequence):
    """Class for a sequence of pitch-yaw images

    Parameters
    ----------
    list_of_files : list of str
        List of FITS files with pitch-yaw images
    """
    def __init__(self, list_of_files):
        FrameSequence.__init__(self, list_of_files, frame_class=PYFrame)

    @property
    def dataframe(self):
        """Obtain a pandas DataFrame"""
        center = [x.peaks[0] for x in self]
        return pd.DataFrame({'center_x' : [z[1] for z in center],
                             'center_y' : [z[0] for z in center]},
                            index=oeb2utc([x.trigger_time for x in self]))

    def plot_centers(self, **dataframe_plot_kwargs):
        """Plot the center of the brightest Sun across the entire image sequence"""
        return self.dataframe.plot(**dataframe_plot_kwargs)


class RSequence(FrameSequence):
    """Class for a sequence of roll images

    Parameters
    ----------
    list_of_files : list of str
        List of FITS files with roll images
    """
    def __init__(self, list_of_files):
        FrameSequence.__init__(self, list_of_files, frame_class=RFrame)

    @property
    def dataframe(self):
        """Obtain a pandas DataFrame"""
        return pd.DataFrame({'midpoint' : [x.midpoint for x in self]},
                            index=oeb2utc([x.trigger_time for x in self]))

    def plot_midpoint(self, **dataframe_plot_kwargs):
        """Plot the midpoints"""
        return self.dataframe.plot(**dataframe_plot_kwargs)
