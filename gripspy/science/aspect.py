from __future__ import division, absolute_import, print_function

from sys import stdout

import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import match_template, peak_local_max
from skimage.transform import warp
from astropy.io import fits


__all__ = ['PYFrame', 'PYSequence', 'RFrame', 'RSequence']


NUM_ROWS = 960
NUM_COLS = 1280


def _kernel(radius):
    axis = np.arange(-(radius * 2), radius * 2 + 1)
    x, y = np.meshgrid(axis, axis)
    template = 1 - np.cosh(np.sqrt(x**2 + y**2) * 5. / radius) / 200.
    template[template < 0] = 0
    return template


template_sun = _kernel(50)
template_fiducial = 1 - _kernel(4)


class Frame(object):
    def __init__(self, filename, uid=None, decimation_range=(2, 10)):
        with fits.open(filename) as f:
            if uid != None and f[0].header['CAMERA'] != uid:
                raise RuntimeError
            self.header = f[0].header
            self.data = f[1].data.astype(np.uint8)
            self.trigger_time = f[0].header['GT_TRIG']
        self.decimation = self._detect_decimation(decimation_range)
        self.data = self.data[self.decimation[1]::self.decimation[0], :]

        if self.decimation == (1, 0):
            self.image = self.data
        else:
            transformation = np.matrix([[1, 0, 0],
                                        [0, 1. / self.decimation[0], self.decimation[1] / self.decimation[0]],
                                        [0, 0, 1]])
            self.image = warp(self.data, transformation, output_shape=(NUM_ROWS, NUM_COLS))

    def _detect_decimation(self, decimation_range):
        dead1 = (self.data == 0).sum(1) == NUM_COLS
        for step in range(decimation_range[0], decimation_range[1] + 1):
            sub_rows = NUM_ROWS // step
            dead2 = dead1[:sub_rows * step].reshape(sub_rows, step).sum(0) == sub_rows
            idx = np.flatnonzero(~dead2)
            if len(idx) == 1:
                return (step, idx[0])
        return (1, 0)

    def plot_image(self, **kwargs):
        imshow_args = {'cmap'   : 'gray',
                       'extent' : [-0.5, NUM_COLS - 0.5, NUM_ROWS - 0.5, -0.5],
                       'interpolation' : 'none'}
        imshow_args.update(kwargs)
        plt.imshow(self.data, **imshow_args)


class PYFrame(Frame):
    def __init__(self, filename, uid=158434):
        try:
            Frame.__init__(self, filename, uid=uid)
        except RuntimeError:
            raise RuntimeError("This file does not appear to be a valid frame from the pitch-yaw camera")

        self.peaks, self.fiducials = self._process()

    def _process(self):
        # Perform a coarse search for Suns
        coarse_image = self.image[::10, ::10]
        coarse_match = match_template(coarse_image, template_sun[::10, ::10], pad_input=True)
        coarse_peaks = peak_local_max(coarse_match, threshold_abs=0.9, num_peaks=3)

        fine_peaks = []
        strength = []
        fiducials = []

        for coarse_peak in coarse_peaks:
            # For each coarse detection, do a detection at the full resolution
            sub_image = self.image[coarse_peak[0] * 10 - 110:coarse_peak[0] * 10 + 111,
                                   coarse_peak[1] * 10 - 110:coarse_peak[1] * 10 + 111]
            match = match_template(sub_image, template_sun, pad_input=True)
            peak = peak_local_max(match, threshold_abs=0.9, num_peaks=1)
            if len(peak) > 0:
                peak = peak[0]
                peak += coarse_peak * 10 - 110

                fine_peaks.append(tuple(peak))

                #FIXME: need a more robust estimate of the strength of each peak
                strength.append(self.image[peak[0], peak[1]])

                # Find fiducials near the center of the Sun
                match = match_template(self.image[peak[0]-60:peak[0]+61, peak[1]-60:peak[1]+61],
                                       template_fiducial, pad_input=True)
                fids = peak_local_max(match, threshold_abs=0.8)
                for fid in fids:
                    fid += peak - 60
                    fiducials.append(tuple(fid))

        # Sort the peaks in order of decreasing strength
        fine_peaks = [peak for (strength, peak) in sorted(zip(strength, fine_peaks), reverse=True)]

        return fine_peaks, fiducials

    def plot_image(self, **kwargs):
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

        Frame.plot_image(self, **kwargs)


class RFrame(Frame):
    def __init__(self, filename, uid=142974):
        try:
            Frame.__init__(self, filename, uid=uid, decimation_range=(10, 10))
        except RuntimeError:
            raise RuntimeError("This file does not appear to be a valid frame from the roll camera")


class FrameSequence(list):
    pass


class PYSequence(FrameSequence):
    def __init__(self, list_of_files):
        FrameSequence.__init__(self)

        for entry in list_of_files:
            self.append(PYFrame(entry))
            stdout.write(".")

        stdout.write("\n")

    def plot_centers(self, **kwargs):
        time = []
        center_y = []
        center_x = []
        for frame in self:
            if frame.peaks:
                time.append(frame.trigger_time)
                center_x.append(frame.peaks[0][1])
                center_y.append(frame.peaks[0][0])
        plt.plot(time, center_x, label='X (pixels)')
        plt.plot(time, center_y, label='Y (pixels)')


class RSequence(FrameSequence):
    def __init__(self, list_of_files):
        FrameSequence.__init__(self)

        for entry in list_of_files:
            self.append(RFrame(entry))
