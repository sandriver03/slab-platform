import labplatform.core.TDTblackbox as tdt

import numpy as np
import os
import importlib
import sys
import scipy.signal as ssig
import matplotlib.pyplot as plt


def get_sound_speed(temp=20.):
    """
    speed of sound at given temperature in celsius
    Args:
        temp: float, celsius

    Returns:
        float
    """
    return 331. + 0.6 * temp


def get_system_delay(speaker_distance=0.1, temperature=20.0, sampling_freq=97656.25,
                     play_device='RX8', rec_device='RX6'):
    """
    Calculate the delay it takes for played sound to be recorded. Depends
    on the distance of the microphone from the speaker and on the devices
    digital-to-analog and analog-to-digital conversion delays.

    Args:
        speaker_distance: float, in meter
        temperature: float, room temperature in celsius
        sampling_freq: Hz
        play_device: str, processor class of the playing device
        rec_device: str

    Returns:
        int, number of samples
    """
    return int(speaker_distance / get_sound_speed(temperature) * sampling_freq) + \
           tdt.get_dac_delay(play_device) + tdt.get_adc_delay(rec_device)


def dB_to_Vrms(dB, calibration_val, calibration_Vrms=1.0):
    """
    convert dB for sound pressure level to voltage amplitude to be used in TDT processor
    Args:
        dB: target dB value
        calibration_val: calibration dB value for the setup
        calibration_Vrms: Vrms where the calibration value is obtained. should be 1.0 by default

    Returns:
        corresponding voltage amplitude
    """
    calibration_val = calibration_val + 20.0 * np.log10(1 / 2e-5)
    return calibration_Vrms * 10 ** ((dB - calibration_val) / 20.0)


def dBdiff_to_Vrmsratio(dBdiff):
    """
    convert dB differences (+-) into voltage amplitude ratios
    :param dBdiff: float, array-like
    :return:
    """
    if isinstance(dBdiff, (tuple, list)):
        dBdiff = np.array(dBdiff)
    return 10 ** (dBdiff / 20.)


def Vrmsratio_to_dBdiff(Vrmsratio):
    """
    convert voltage amplitude ratios into dB differences (+-)
    :param Vrmsratio: float, array-like
    :return:
    """
    if isinstance(Vrmsratio, (tuple, list)):
        Vrmsratio = np.array(Vrmsratio)
    return 20 * np.log10(Vrmsratio)


def Vrms_to_dB(Vrms, calibration_val):
    """
    convert voltage amplitude to dB for sound pressure level to be used in TDT processor
    Args:
        Vrms: voltage amplitude in TDT processor
        calibration_val: calibration dB value for the setup

    Returns:
        corresponding dB value
    """
    return 20.0 * np.log10(Vrms / 2e-5) + calibration_val


def importfile(path):
    """Import a Python source file or compiled file given its path."""
    magic = importlib.util.MAGIC_NUMBER
    with open(path, 'rb') as file:
        is_bytecode = magic == file.read(len(magic))
    filename = os.path.basename(path)
    name, ext = os.path.splitext(filename)
    if is_bytecode:
        loader = importlib._bootstrap_external.SourcelessFileLoader(name, path)
    else:
        loader = importlib._bootstrap_external.SourceFileLoader(name, path)
    # XXX We probably don't need to pass in the loader here.
    spec = importlib.util.spec_from_file_location(name, path, loader=loader)
    try:
        return importlib._bootstrap._load(spec)
    except:
        raise ErrorDuringImport(path, sys.exc_info())


class ErrorDuringImport(Exception):
    """Errors that occurred while trying to import something to document it."""
    def __init__(self, filename, exc_info):
        self.filename = filename
        self.exc, self.value, self.tb = exc_info

    def __str__(self):
        exc = self.exc.__name__
        return 'problem in %s - %s: %s' % (self.filename, exc, self.value)


def filter_freq_response(slab_filter, axes=None, freq_range=(2000, 46000)):
    """
    plot the filter frequency response. currently only consider the FIR filter
    :param slab_filter: slab.Filter instance
    :param axes: matplotlib axes object, if provided, will plot on this axes
    :param freq_range: tuple, x axis range
    :return: axes object
    """
    # w is in radian/sample, need to convert to hertz
    w, h = ssig.freqz(slab_filter.data, fs=slab_filter.samplerate)
    if axes is None:
        fig, ax1 = plt.subplots()
    else:
        ax1 = axes
    ax1.set_title('Digital filter frequency response')
    ax1.semilogx(w, 20 * np.log10(abs(h)))
    ax1.set_ylabel('Amplitude [dB]')
    ax1.set_xlabel('Frequency [Hz]')
    ax1.set_xlim(freq_range)
    return ax1


def shuffle_filter_coefs(coefs_array, dtype=np.float32):
    """
    shuffle the filter coefficient array, so it can be used in the TDT FIR2 component
    see TDT RPvdsEx manual for the details
    :param coefs_array: 2d coefficient array, 2nd dimension is channel
    :param dtype: data type of returned array
    :return: shuffled and flattened coefficient array
    """
    if len(coefs_array.shape) < 2:
        coefs_array = coefs_array.reshape(-1, 1)
    # 1st dimension is the taps and 2nd the channels
    n_taps = coefs_array.shape[0]
    if n_taps % 2 != 0:
        raise ValueError("the nTaps of the filters must be even")
    # set data type to float32, which is used in TDT
    coefs_shuffled = np.zeros(coefs_array.shape, dtype=dtype)
    coefs_shuffled[::2, :] = coefs_array[:int(n_taps/2), :]
    coefs_shuffled[1::2, :] = coefs_array[int(n_taps/2):, :]
    return coefs_shuffled.ravel(order='F')


def prepare_hwfilter_coefs(spks):
    """
    given a list of speakers, prepare the hardware filter coefs for the TDT
    :param spks: list of speakers
    :return: np array, int
    """
    ntaps = spks[0].filter.data.shape[0]
    ids = [spk.id for spk in spks]
    coefs = np.zeros((ntaps, max(ids) + 1), dtype=np.float32)
    for spk in spks:
        coefs[:, spk.id] = spk.filter_hardware.data.flatten()
    return shuffle_filter_coefs(coefs), ntaps

