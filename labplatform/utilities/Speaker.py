import slab
from labplatform.config import get_config
import labplatform.core.TDTblackbox as tdt

from copy import deepcopy
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import time
import logging
log = logging.getLogger(__name__)


_rec_tresh = 65  # treshold in dB above which recordings are not rejected
_speaker_configs = None


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
                     play_device='RX6', rec_device='RX6'):
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


def load_config(config=None):
    """
    load pre-defined speaker calibration configurations
    Args:
        config: str, name of the config to be loaded. if None, return all the configs defined

    Returns:
        dict
    """
    global _speaker_configs
    if not _speaker_configs:
        from labplatform.config import SpeakerCalibration_config as settings
        setting_names = [s for s in dir(settings) if s.upper() == s or s == '__spec__']
        setting_values = [getattr(settings, s) for s in setting_names]
        _speaker_configs = dict(zip(setting_names, setting_values))

    if config is not None:
        return _speaker_configs[config]
    else:
        return _speaker_configs


def initialize_devices(config):
    """
    initialize all devices included in config
    Args:
        config: dictionary

    Returns:
        dict of initialized devices
    """
    devices = {}
    if config['ZBus']['use_ZBus']:
        devices['ZBus'] = tdt.initialize_zbus(config['ZBus']['connection'])
    else:
        devices['ZBus'] = None
    rcx_folder = get_config("DEVICE_ROOT")
    for d, v in config['devices'].items():
        devices[d] = tdt.initialize_processor(processor=v['processor'],
                                              connection=v['connection'],
                                              index=v['index'],
                                              path=os.path.join(rcx_folder, v['rcx_file']))
        devices[d].name = d
    return devices


def calibrate_speaker_SPL(speaker_name, stim_device, sys_config, file_name=None):
    """
    calibrate speaker loudness level for
    Args:
        speaker_name: str, name of the speaker to be calibrated
        stim_device: tdt processor handle, used to generate 1kz sound output
        sys_config: str, name of the system to be calibrated
        file_name: str, name of the calibration file to be saved

    Returns:
        None
    """
    # play 1k Hz pure tone from the speaker
    print('Playing 1kHz test tone from the system. Please measure intensity.')
    stim_device.SoftTrg(1)
    intensity = float(input('Enter measured intensity in dB: '))  # ask for measured intesnity
    # subtract rms from measured intensity
    intensity = intensity - 20.0*np.log10(1/2e-5)
    # generate file name if none
    if file_name is None:
        import datetime
        date_str = datetime.datetime.now().strftime(get_config("DATE_FMT"))
        file_name = "SPL_{}_{}_{}".format(speaker_name, sys_config, date_str)
    folder = get_config('CAL_ROOT')
    # save result as npz file
    abs_fname = os.path.join(folder, file_name)
    if os.path.exists(abs_fname):
        print("file {} already exists. Overriding it...".format(abs_fname))
    np.save(abs_fname, intensity)
    # stop the sound
    stim_device.SoftTrg(1)


def foo_calibrate_speaker_SPL(speaker_name, stim_device=None, sys_config='Testing', file_name=None):
    """
    dummy function, simulate calibrating speaker loudness level without hardwares. for testing
    Args:
        speaker_name: str, name of the speaker to be calibrated
        stim_device: tdt processor handle, used to generate 1kz sound output
        sys_config: str, name of the system to be calibrated
        file_name: str, name of the calibration file to be saved

    Returns:
        None
    """
    # play 1k Hz pure tone from the speaker
    print('Playing 1kHz test tone from the system. Please measure intensity.')
    # stim_device.SoftTrg(1)
    intensity = float(input('Enter measured intensity in dB: '))  # ask for measured intesnity
    # subtract rms from measured intensity
    intensity = intensity - 20.0*np.log10(1/2e-5)
    # generate file name if none
    if file_name is None:
        import datetime
        date_str = datetime.datetime.now().strftime(get_config("DATE_FMT"))
        file_name = "SPL_{}_{}_{}".format(speaker_name, sys_config, date_str)
    folder = get_config('CAL_ROOT')
    # save result as npz file
    abs_fname = os.path.join(folder, file_name)
    if os.path.exists(abs_fname):
        print("file {} already exists. Overriding it...".format(abs_fname))
    np.save(abs_fname, intensity)
    # stop the sound
    # stim_device.SoftTrg(1)


def load_SPL_calibration(speaker, sys_config, load_latest=True):
    """
    load speaker SPL calibration result
    Args:
        speaker: str, name of the speaker
        sys_config: str, name of the system
        load_latest: bool, if multiple files exist only load latest one; otherwise load all

    Returns:
        calibration result; single number, or a dictionary (if multiple files, and load_latest is false)
    """
    cal_path = get_config('CAL_ROOT')
    fname_ = 'SPL_{}_{}_*.npy'.format(speaker, sys_config)
    import glob
    files = glob.glob(os.path.join(cal_path, fname_))
    if not files:
        raise ValueError("speaker {} in system {} has not been calibrated for SPL".format(speaker, sys_config))
    elif files.__len__() == 1:
        return np.load(files[0])
    else:
        if load_latest:
            return np.load(files[-1])
        else:
            return {os.path.basename(f): np.load(f) for f in files}


def calibrate_speaker_frequency(speaker, sys_config, bandwidth,
                                low_lim, hi_lim, alpha):
    """
    play the level-equalized signal, record and compute and a bank of inverse filter
    to equalize each speaker relative to the target one. Return filterbank and recordings

    Args:
        speaker: str, name of the speaker
        sys_config: str, name of the system
        bandwidth:
        low_lim:
        hi_lim:
        alpha:
        exclude:

    Returns:

    """
    pass


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


def equalize_speaker_SPL_fromfile(speaker_list, target_speaker, sys_config):
    """
    get a ratio of Vrms in speaker_list to target speaker so the output SPLs are the same
    Args:
        speaker_list: tuple or list, name of speakers to calculate the ratios
        target_speaker: str, name of the target speaker
        sys_config: str, system configuration name

    Returns:
        np.array of float, gain relative to target speaker
    """
    # load speaker specific calibration value
    Lcal = []
    for speaker in speaker_list:
        Lcal.append(load_SPL_calibration(speaker, sys_config))
    Ltar = load_SPL_calibration(target_speaker, sys_config)
    return 10 ** ((np.array(Lcal) - Ltar) / 20.0)


def play_and_record(sig, play_device, rec_device, trigger_params,
                    recdata_tags=('audio_data', ),
                    len_tags=('stimBufLen_n', 'recBufLen_n'),
                    config_info={'distance': 0.15, 'temperature': 20.0},
                    ):
    """
    play the signal from a given device and get recorded response from the device
    Args:
        sig: slab.Sound instance, signal to be played
        play_device: the device used to play the signal
        rec_device: the device used to record the response
        trigger_params: dict, how to trigger the entire process. see tdt.trigger()
        recdata_tags: tuple or list of str; names of data tags to be read
        len_tags: tuple of str, tags of stimulus length and recording length to be set
        config_info: additional system configuration information

    Returns:
        list of np.array(s)
    """
    # check if sampling rates match
    if abs(sig.samplerate - play_device.GetSFreq()) > 1:
        raise ValueError(f"sample rate of the signal: {sig.samplerate} does not match that of the "
                         f"playing device: {play_device.GetSFreq()}")
    # load the signal into playing device
    tdt.set_variable('stim_data', sig.data, play_device)
    # set appropriate playing and recording length
    # no need to consider play device, only adjust it in recording device
    delay_n = get_system_delay(sampling_freq=rec_device.GetSFreq(),
                               play_device=play_device.processor,
                               rec_device=rec_device.processor,
                               temperature=config_info['temperature'],
                               speaker_distance=config_info['distance'])
    play_device.SetTagVal(len_tags[0], sig.nsamples)
    rec_device.SetTagVal(len_tags[1], sig.nsamples + delay_n)
    # send trigger
    if isinstance(trigger_params, dict):
        tdt.trigger(**trigger_params)
    elif isinstance(trigger_params, (list, tuple)):
        tdt.trigger(*trigger_params)
    # wait for finish
    wait_to_finish_playing(rec_device, )
    # read recording buffer from recording device and return it
    res = []
    for tag in recdata_tags:
        nsamples = rec_device.GetTagVal(tag + '_i')
        res.append(rec_device.ReadTagV(rec_device, 0, nsamples))
    return res


def wait_to_finish_playing(proc, tagname="recording"):
    """
    Busy wait as long as sound is played from the processors. The .rcx file
    must contain a tag (default name is playback) that has the value 1 while
    sound is being played and 0 otherwise.

    Args:
        proc: TDT handle, processor in which the tag is checked
        tagname: str, name of the tag to be checked

    Returns:

    """
    log.info(f'Waiting for {tagname} on {proc}.')
    while proc.GetTagVal(tagname):
        time.sleep(0.01)
    log.info('Done waiting.')


# it probably make sense to use pandas to hold speaker information, as used in the freefield toolbox
class Speakers:
    """
    hold information for a group of speakers
    by default, the configuration file should be in the CAL_ROOT
    """
    __speaker_table_cols = {'name': 'object',
                            'id': np.int,
                            'channel': np.int,  # should be a digital bit in the device
                            'azi': np.float,  # azimuth angle, degree
                            'ele': np.float,  # elevation, degree
                            'dis': np.float,  # distance, m
                            'device_name': 'object',  # should be key in a dict linked to the device handel
                            'addr': np.int,  # similar to channel, a digital address on the device
                            }

    def __init__(self, file=None, speakers=None, setup=None, ):
        """
        Args:
            file: str or path
            speakers: pandas.DataFrame
            setup: str, name of the setup
        """
        self.speakers = speakers
        self.setup = setup
        # file should be abs path; if not, use form '{setup}_speakers.csv'
        if file is not None:
            self.speaker_file = self._filename_from_file(file)
        elif setup is not None:
            self.speaker_file = self._filename_from_setup(setup)
        self.speaker_file = file

    def _filename_from_file(self, file):
        """
        get the absolute file path to the speaker configuration file. the configuration file is
        put into get_config('CAL_ROOT')
        Args:
            file: str or path
        Returns:
            absolute file path
        """
        if not os.path.isabs(file):
            file = os.path.join(get_config('CAL_ROOT'), file)
            if not file.endswith('.csv'):
                file = file + '.csv'
        return file

    def _filename_from_setup(self, setup):
        """
        get the absolute file path to the speaker configuration file. file name in form of:
            '{setup}_speakers.csv'
        the configuration file is put into get_config('CAL_ROOT')
        Args:
            setup: str, name of the setup
        Returns:
            absolute file path
        """
        return os.path.join(get_config('CAL_ROOT'), '{}_speakers.csv'.format(setup))

    def add_speaker(self, **kwargs):
        """
        add a speaker to the speaker list (in pd.DataFrame)
        Args:
            **kwargs: see self.__speaker_table_cols
        Returns:
            None
        """
        # create an empty pd.DataFrame with the same columns as self.speakers
        new_speaker = self.speakers.loc[[False] * len(self.speakers)]
        for k, v in kwargs:
            if k in self.__speaker_table_cols.keys():
                new_speaker[k] = v
        # index of the speaker can be automatically generated if not given
        if new_speaker['id'] is None:
            new_speaker['id'] = self.speakers['id'].max() + 1
        # append the new speaker to the speaker list
        self.speakers.append(new_speaker, ignore_index=True)

    def remove_speaker(self, **kwargs):
        """
        remove a speaker from the speaker list (in pd.DataFrame)
        Args:
            **kwargs: used to identify the speaker to be removed, only 'name', 'id', 'ele', 'azi'
                      are used. 'ele' and 'azi' must be used in pairs
        Returns:
            None
        """
        pass

    def save(self, file=None, overwrite=True):
        """
        save current speakers information into a .csv file
        Args:
            file: str or path
            overwrite: bool, if overwrite existing file
        """
        if file is None:
            file = self.speaker_file
        if not overwrite and os.path.isfile(file):
            raise RuntimeError('the speaker file: {} already exists!'.format(file))
        # save self.speakers into a .csv file
        self.speakers.to_csv(file, index=False)

    def load(self, file=None):
        """
        load speaker information from a file
        Args:
            file: str or path, if None use self.speaker_file
        """
        if file is None:
            file = self.speaker_file
        # load .csv file using Pandas
        self.speakers = pd.read_csv(file, dtype=self.__speaker_table_cols)


def _level_equalization(sig, all_speakers, target_index):
    """
    Record the signal from each speaker in the list and return the level of each
    speaker relative to the target speaker
    Args:
        sig:
        all_speakers: pd data frame contains information for all speakers
        target_index: index of the target speaker in the pd frame

    Returns:

    """
    # TODO: probably need a class for speakers, which contains:
    #   device handle for the speaker (?)
    #   device name for the speaker, in accordance with the configuration
    #   name of the speaker
    #   index of the speaker
    #   channel of the speaker
    #   location of the speaker
    #   relative SPL factor
    rec = []
    for row in all_speakers:
        rec.append(play_and_record(row[0], sig, apply_calibration=False))

    rec = slab.Sound(rec)
    rec.data[:, rec.level < _rec_tresh] = rec[target_index].data  # thresholding
    return rec[target_index].level / rec.level


def _frequency_equalization(sig, speaker_list, target_speaker, bandwidth,
                            low_lim, hi_lim, alpha, exclude=None):
    """
    play the level-equalized signal, record and compute and a bank of inverse filter
    to equalize each speaker relative to the target one. Return filterbank and recordings
    """
    rec = []
    for row in speaker_list:
        modulated_sig = deepcopy(sig)  # copy signal and correct for lvl difference
        modulated_sig.level *= _level_equalization(sig, speaker_list, target_speaker)[int(row[0])]
        rec.append(play_and_record(row[0], modulated_sig, apply_calibration=False))
        if row[0] == target_speaker:
            target = rec[-1]
    rec = slab.Sound(rec)
    # set recordings which are below the threshold or which are from exluded speaker
    # equal to the target so that the resulting frequency filter will be flat
    rec.data[:, rec.level < _rec_tresh] = target.data
    if exclude is not None:
        for e in exclude:
            print("excluding speaker %s from frequency equalization..." % (e))
            rec.data[:, e] = target.data[:, 0]
    fbank = slab.Filter.equalizing_filterbank(target=target, signal=rec, low_cutoff=low_lim,
                                              high_cutoff=hi_lim, bandwidth=bandwidth, alpha=alpha)
    # check for notches in the filter:
    dB = fbank.tf(plot=False)[1][0:900, :]
    if (dB < -30).sum() > 0:
        print("WARNING! The filter contains large notches! You might want to adjust the \n"
               " alpha parameter or set plot=True to see the speakers affected...")
    return fbank, rec


def spectral_range(signal, bandwidth=1/5, low_lim=50, hi_lim=24000, thresh=3,
                   plot=True, log_scale=True):
    """
    Compute the range of differences in power spectrum for all channels in
    the signal. The signal is divided into bands of equivalent rectangular
    bandwidth (ERB - see More&Glasberg 1982) and the level is computed for
    each frequency band and each channel in the recording. To show the range
    of spectral difference across channels the minimum and maximum levels
    across channels are computed. Can be used for example to check the
    effect of loud speaker equalization.
    """
    # TODO: this really should be part of the slab.Sound file
    # generate ERB-spaced filterbank:
    fbank = slab.Filter.cos_filterbank(length=1000, bandwidth=bandwidth,
                                       low_cutoff=low_lim, high_cutoff=hi_lim,
                                       samplerate=signal.samplerate)
    center_freqs, _, _ = slab.Filter._center_freqs(low_lim, hi_lim, bandwidth)
    center_freqs = slab.Filter._erb2freq(center_freqs)
    # create arrays to write data into:
    levels = np.zeros((signal.nchannels, fbank.nchannels))
    max_level, min_level = np.zeros(fbank.nchannels), np.zeros(fbank.nchannels)
    for i in range(signal.nchannels):  # compute ERB levels for each channel
        levels[i] = fbank.apply(signal.channel(i)).level
    for i in range(fbank.nchannels):  # find max and min for each frequency
        max_level[i] = max(levels[:, i])
        min_level[i] = min(levels[:, i])
    difference = max_level-min_level
    if plot is True or isinstance(plot, Axes):
        if isinstance(plot, Axes):
            ax = plot
        else:
            fig, ax = plt.subplots(1)
        # frequencies where the difference exceeds the threshold
        bads = np.where(difference > thresh)[0]
        for y in [max_level, min_level]:
            if log_scale is True:
                ax.semilogx(center_freqs, y, color="black", linestyle="--")
            else:
                ax.plot(center_freqs, y, color="black", linestyle="--")
        for bad in bads:
            ax.fill_between(center_freqs[bad-1:bad+1], max_level[bad-1:bad+1],
                            min_level[bad-1:bad+1], color="red", alpha=.6)
    return difference


def _plot_equalization(target, signal, filt,
                       speaker_name,
                       sys_config,
                       low_lim=50,
                       hi_lim=20000,
                       bandwidth=1/8,
                       save_fig=True):
    """
    Make a plot to show the effect of the equalizing FIR-filter on the
    signal in the time and frequency domain. The plot is saved to the log
    folder (existing plots are overwritten)

    Args:
        target: target signal to be compared with, slab.sound.Sound instance
        signal: signal to be filtered, slab.sound.Sound instance
        filt: equalization filter, slab.filter.Filter instance
        speaker_name: str, name of the speaker where signal is played
        sys_config: str, name of the configuration the speaker belongs
        low_lim:
        hi_lim:
        bandwidth:
        save_fig: bool, if save figure to calibration folder

    Returns:

    """
    signal_filt = filt.apply(signal)  # apply the filter to the signal
    fig, ax = plt.subplots(2, 2, figsize=(16., 8.))
    fig.suptitle("Equalization Speaker %s " % speaker_name)
    ax[0, 0].set(title="Power per ERB-Subband", ylabel="A")
    ax[0, 1].set(title="Time Series", ylabel="Amplitude in Volts")
    ax[1, 0].set(title="Equalization Filter Transfer Function",
                 xlabel="Frequency in Hz", ylabel="Amplitude in dB")
    ax[1, 1].set(title="Filter Impulse Response",
                 xlabel="Time in ms", ylabel="Amplitude")
    # get level per subband for target, signal and filtered signal
    fbank = slab.Filter.cos_filterbank(
        1000, bandwidth, low_lim, hi_lim, signal.samplerate)
    center_freqs, _, _ = slab.Filter._center_freqs(low_lim, hi_lim, bandwidth)
    center_freqs = slab.Filter._erb2freq(center_freqs)
    for data, name, color in zip([target, signal, signal_filt],
                                 ["target", "signal", "filtered"],
                                 ["red", "blue", "green"]):
        levels = fbank.apply(data).level
        ax[0, 0].plot(center_freqs, levels, label=name, color=color)
        ax[0, 1].plot(data.times*1000, data.data, alpha=0.5, color=color)
    ax[0, 0].legend()
    w, h = filt.tf(plot=False)
    ax[1, 0].semilogx(w, h, c="black")
    ax[1, 1].plot(filt.times, filt.data, c="black")
    # figure should be saved to CAL_ROOT
    fname = 'speakerCalRes_{}_{}.png'.format(speaker_name, sys_config)
    if save_fig:
        fig.savefig(os.path.join(get_config('CAL_ROOT'), fname), dpi=800)
    plt.close()
