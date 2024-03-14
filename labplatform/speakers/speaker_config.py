from labplatform.config import get_config
import slab

from dataclasses import dataclass
from copy import deepcopy
import os
import numpy as np
# import pandas as pd
import json
import fnmatch
import re
import datetime
import pickle


_speaker_configs = None

SPEAKER_TABLE_COLS = [('name', str),
                      ('id', int),
                      ('azimuth', float),  # azimuth angle, degree
                      ('elevation', float),  # elevation, degree
                      ('distance', float),  # distance, m
                      ('channel_analog', int),  # the analog channel number the speaker is connected to
                      ('TDT_analog', str),  # name of the TDT device used to control this speaker
                      ('TDT_idx_analog', int),  # index of the TDT device used to control this speaker
                      ('channel_digital', int),  # the digital channel number the speaker is connected to
                      ('TDT_digital', str),  # name of the TDT device used to control this speaker
                      ('TDT_idx_digital', int),  # index of the TDT device used to control this speaker
                      ]


def validate_config(config_dict):
    """
    check if required information is present in the speaker configuration definitions
    the configuration should contain:
        1. speaker_file: str, name/path to the speaker info file. if not an absolute path, then assume it is in the
        CAL_ROOT folder of the labplatform
        2. device: dict, points to the Device class used in the calibration; should contain:
            a. file_name: str, .py file defining the class; if not an absolute path, then assumed in the DEVICE_ROOT
                it can also be a package.module str
            b. device_class: str, name of the class
        3. TDT_rec: str, TDT processor type used to do recording
        4. TDT_aud: str, TDT processor type used to play the stimulus
        5. ref_spk_id: optional, id of the reference speaker
    :param config_dict: dict containing the configuration definitions
    :return:
    """
    # TODO


def load_speaker_config(config=None):
    """
    load pre-defined speaker calibration configurations
    custom config file should be json files named {config}_speaker_config.json
    the configuration contains:
        1. speaker_file: str, speaker table file name
        2. device: dict of strs, device class used to perform the calibration
            file_name, name of the .py file
            class_name, name of the device class
    Args:
        config: str, name of the config to be loaded. if None, return the default configs defined

    Returns:
        dict
    """
    global _speaker_configs
    # load default configs into the global
    if not _speaker_configs:
        from . import default_config as speaker_setting
        setting_names = [s for s in dir(speaker_setting) if s.upper() == s or s == '__spec__']
        setting_values = [getattr(speaker_setting, s) for s in setting_names]
        _speaker_configs = dict(zip(setting_names, setting_values))

    if config is not None:
        if config in _speaker_configs:
            return _speaker_configs[config]
        else:
            # try to load non-default config specified
            file_dir = get_config("CAL_ROOT")
            fname = "{}_speaker_config.json"
            with open(os.path.join(file_dir, fname), 'r') as fh:
                params = json.load(fh)
            # update global
            _speaker_configs[config] = params
            return params
    else:
        return _speaker_configs


def save_speaker_config(config, param_dict):
    """
    save speaker configuration parameters into a json file
    :param config: str, name of the config
    :param param_dict: dict containing two keys, 'speaker_file' and 'device'
    :return:
    """
    if 'speaker_file' not in param_dict:
        raise ValueError('the configuration parameters must contain the name of the speaker table file')
    if 'device' not in param_dict:
        raise ValueError('the configuration parameters must contain the name of the device')
    file_dir = get_config("CAL_ROOT")
    fname = "{}_speaker_config.json"
    with open(os.path.join(file_dir, fname), 'w') as fh:
        json.dump(param_dict, fh)


@dataclass
class Speaker:
    """
    Class for handling the loudspeakers which are usually loaded using `read_speaker_table().`.
    """
    name: str = None  # name of the speaker
    id: int = None  # the id of the speaker, need to be unique
    azimuth: float = None  # the azimuth angle of the speaker
    elevation: float = None  # the azimuth angle of the speaker
    distance: float = None  # distance of the speaker to subject, m
    channel_analog: int = None  # the number of the analog channel to which the speaker is attached
    TDT_analog: str = None  # the name of the processor to whose analog I/O the speaker is attached
    TDT_idx_analog: str = None  # the index of the processor to whose analog I/O the speaker is attached
    channel_digital: int = None  # the number of the digital channel to which the speaker is attached
    TDT_digital: str = None  # the name of the processor to whose digital I/O the speaker is attached
    TDT_idx_digital: str = None  # the index of the processor to whose digital I/O the speaker is attached

    level: float = None  # sound level difference w.r.t. the reference speaker, in dB. +-, not ratio
    filter: slab.Filter = None  # filter for equalizing the filters transfer function, when used in software
    filter_hardware: slab.Filter = None  # filter to be used in hardware (e.g. TDT)
    calib_dB: float = None  # sound level at which this speaker is calibrated

    def __repr__(self):
        if (self.level is None) and (self.filter is None):
            calibrated = "NOT calibrated"
        else:
            calibrated = "calibrated"
        return f"<speaker {self.id} at azimuth {self.azimuth}, elevation {self.elevation}, " \
               f"distance {self.distance}, {calibrated}>"

    def attr_str(self):
        s_None = ','.join([str(getattr(self, a[0])) for a in SPEAKER_TABLE_COLS])
        s = s_None.replace('None', '')
        return s

    def TDT_name_analog(self):
        return '{}{}'.format(self.TDT_analog, self.TDT_idx_analog)

    def TDT_name_digital(self):
        return '{}{}'.format(self.TDT_digital, self.TDT_idx_digital)

    @staticmethod
    def header_str():
        return ','.join([a[0] for a in SPEAKER_TABLE_COLS])

    def apply_equalization(self, signal, level_only=True):
        """
        Apply level correction and frequency equalization to a signal
        Args:
            signal: signal to calibrate
            level_only: bool, if apply level equalization only
        Returns:
            slab.Sound: calibrated copy of signal
        """
        signal = slab.Sound(signal)
        equalized_signal = deepcopy(signal)
        if level_only:
            if self.level is None:
                raise ValueError("speaker not level-equalized! Load an existing "
                                 "equalization of calibrate the setup!")
            equalized_signal.level -= self.level
            return equalized_signal
        else:
            if self.filter is None:
                raise ValueError("speaker not frequency-equalized! Load an existing "
                                 "equalization of calibrate the setup!")
            equalized_signal = self.filter.apply(equalized_signal)
            return equalized_signal


# it probably makes sense to use pandas to hold speaker information, as used in the freefield toolbox
class SpeakerArray:
    """
    hold information for a group of speakers
    by default, the configuration file should be in the CAL_ROOT
    to find individual speaks, see pandas.dataFrame
    file name for the speakers should always be in form of {setup}_speakers.txt
    """
    # however, we still need additional information for each speaker:
    #   1. if this speaker is used as reference
    #   2. amplitude adjustment for each speaker w.r.t. the reference
    #   3. filter bank for each speaker w.r.t. the reference

    def __init__(self, file=None, setup=None, ):
        """
        Args:
            file: str or path
            setup: str, name of the setup
        """
        # Note: the id of the speakers are assumed to be unique
        self.speaker_table = None
        self.ref_spk_id = None
        self.setup = setup
        # file should be abs path; if not, use form '{setup}_speakers.txt'
        if file is not None:
            self.speaker_file = self._filename_from_file(file)
            self.setup = self._filename_to_setup(self.speaker_file)
        elif setup is not None:
            self.speaker_file = self._filename_from_setup(setup)
        else:
            self.speaker_file = None
        # .speakers contains a list of Speaker instances
        self.speakers = None
        # calibration result
        self.calib_result = None

    def _filename_to_setup(self, filename=None):
        """
        infer setup name from file name
        :param filename: str
        :return:
        """
        if filename is None:
            filename = self.speaker_file
        # discard the path
        filename = os.path.split(filename)[-1]
        if filename.find('_speakers.txt') > 0:
            return filename[:filename.find('_speakers.txt')]
        else:
            raise ValueError('File name {} is not valid'.format(filename))

    def _filename_from_file(self, file):
        """
        get the absolute file path to the speaker configuration file. the configuration file is
        put into get_config('CAL_ROOT')
        Args:
            file: str or path
        Returns:
            absolute file path
        """
        # by default, speaker info file is .txt form. if file does not contain extension then .txt is assumed
        if not os.path.splitext(file)[1]:
            file = file + '.txt'
        if not os.path.isabs(file):
            file = os.path.join(get_config('CAL_ROOT'), file)
        return file

    def _filename_from_setup(self, setup):
        """
        get the absolute file path to the speaker configuration file. file name in form of:
            '{setup}_speakers.txt'
        the configuration file is put into get_config('CAL_ROOT')
        Args:
            setup: str, name of the setup
        Returns:
            absolute file path
        """
        return os.path.join(get_config('CAL_ROOT'), '{}_speakers.txt'.format(setup))

    def add_speaker(self, **kwargs):
        """
        add a speaker to the speaker list (in pd.DataFrame)
        Args:
            **kwargs: see SPEAKER_TABLE_COLS
        Returns:
            None
        """
        # check if the new speaker parameters are valid
        self.new_speaker_validation(kwargs)
        # create a speaker instance
        new_speaker = Speaker()
        attr_dict = {a[0]: a[1] for a in SPEAKER_TABLE_COLS}
        for k, v in kwargs:
            if k in attr_dict:
                setattr(new_speaker, k, attr_dict[k](v))
        # index of the speaker can be automatically generated if not given
        if new_speaker.id is None:
            new_speaker.id = max([s.id for s in self.speakers]) + 1
        # validation
        # append the new speaker to the speaker list
        self.speakers.append(new_speaker)

    def new_speaker_validation(self, new_speaker_dict):
        """
        make sure the new speaker is valid, i.e. has a unique id, TDT_name-TDT_idx_channel and azi-ele-dis combination
        :param new_speaker_dict: dict containing the information about the new speaker
        :return:
        """
        # check if id is unique
        if new_speaker_dict['id'] in [s.id for s in self.speakers]:
            raise ValueError('the id for the new speaker is already present')
        # check if position is unique
        if 'azimuth' in new_speaker_dict and 'elevation' in new_speaker_dict and 'distance' in new_speaker_dict:
            pos = [new_speaker_dict['azimuth'], new_speaker_dict['elevation'], new_speaker_dict['distance']]
            pos_spks = [[s.azimuth, s.elevation, s.distance] for s in self.speakers]
            if pos in pos_spks:
                raise Warning('the position for the new speaker is already present in existing speakers')
        # check if analog channel is unique
        if 'channel_analog' in new_speaker_dict and 'TDT_analog' in new_speaker_dict and \
                'TDT_idx_analog' in new_speaker_dict:
            ach = [new_speaker_dict['channel_analog'], new_speaker_dict['TDT_analog'],
                   new_speaker_dict['TDT_idx_analog']]
            ach_spks = [[s.channel_analog, s.TDT_analog, s.TDT_idx_analog] for s in self.speakers]
            if ach in ach_spks:
                raise Warning('the analog control for the new speaker is already present in existing speakers')
        # check if digital channel is unique
        if 'channel_digital' in new_speaker_dict and 'TDT_digital' in new_speaker_dict and \
                'TDT_idx_digital' in new_speaker_dict:
            dch = [new_speaker_dict['channel_digital'], new_speaker_dict['TDT_digital'],
                   new_speaker_dict['TDT_idx_digital']]
            dch_spks = [[s.channel_digital, s.TDT_digital, s.TDT_idx_digital] for s in self.speakers]
            if dch in dch_spks:
                raise Warning('the digital control for the new speaker is already present in existing speakers')

    def check_speaker_params(self):
        """
        check if the speaker table contains duplicates
        need to check:
            1. id should be unique
            2. combination of azi-ele-dis should be unique
            3. combination of TDT_name-TDT_idx-channel should be unique
        :return:
        """
        # check if ids are unique
        if len(set([s.id for s in self.speakers])) < len(self.speakers):
            raise ValueError('the speaker table contains duplicated IDs')
        # check unique positions. ignore those speakers has a None in it
        pos = ['{}_{}_{}'.format(s.azimuth, s.elevation, s.distance) for s in self.speakers
               if s.azimuth and s.elevation and s.distance]
        if len(set(pos)) < len(pos):
            raise Warning('the positions for the speakers are not unique')
        if len(pos) < len(self.speakers):
            raise Warning('the positions for the speakers are not all set')
        # check unique analog controls. ignore those speakers has a None in it
        analog_ch = ['{}_{}_{}'.format(s.channel_analog, s.TDT_analog, s.TDT_idx_analog) for s in self.speakers
                     if s.channel_analog and s.TDT_analog and s.TDT_idx_analog]
        if len(set(analog_ch)) < len(analog_ch):
            raise Warning('the analog controls for the speakers are not unique')
        if len(analog_ch) < len(self.speakers):
            raise Warning('the analog controls for the speakers are not all set')
        # check unique digital controls. ignore those speakers has a None in it
        digital_ch = ['{}_{}_{}'.format(s.channel_digital, s.TDT_digital, s.TDT_idx_digital) for s in self.speakers
                      if s.channel_digital and s.TDT_digital and s.TDT_idx_digital]
        if len(set(digital_ch)) < len(digital_ch):
            raise Warning('the digital controls for the speakers are not unique')
        if len(digital_ch) < len(self.speakers):
            raise Warning('the digital controls for the speakers are not all set')

    def remove_speaker(self, row_idx):
        """
        remove a speaker from the speaker list (in pd.DataFrame)
        Args:
            row_idx: int or list of int, indexes of rows to be removed
        Returns:
            None
        """
        self.speaker_table = self.speaker_table.drop(row_idx)

    def pick_speakers(self, picks):
        """
        Either return the speaker at given coordinates (azimuth, elevation) or the
        speaker with a specific index number.
        Args:
            picks (list of lists, list, int): index number of the speaker
        Returns:
            (list):
        """
        if picks in ('All', 'all', 'ALL'):
            return self.speakers
        if isinstance(picks, (list, np.ndarray)):
            if all(isinstance(p, Speaker) for p in picks):
                spks = picks
            elif all(isinstance(p, (int, np.int64, np.int32)) for p in picks):
                spks = [s for s in self.speakers if s.id in picks]
            else:
                spks = [s for s in self.speakers if (s.azimuth, s.elevation) in picks]
        elif isinstance(picks, (int, np.int64, np.int32)):
            spks = [s for s in self.speakers if s.id == picks]
        elif isinstance(picks, Speaker):
            spks = [picks]
        else:
            spks = [s for s in self.speakers if (s.azimuth == picks[0] and s.elevation == picks[1])]
        if len(spks) == 0:
            print("no speaker found that matches the criterion - returning empty list")
        return spks

    def save_speaker_table(self, file=None, overwrite=True):
        """
        save current speakers information into a .txt file
        Args:
            file: str or path
            overwrite: bool, if overwrite existing file
        """
        if file is None:
            file = self.speaker_file
        if not overwrite and os.path.isfile(file):
            raise RuntimeError('the speaker file: {} already exists!'.format(file))
        # save self.speaker_table into a .txt file
        # sort according to IDs
        spks = sorted(self.speakers, key=lambda x: x.id)
        # write file
        with open(file, 'w') as fh:
            # header line
            fh.writelines(spks[0].header_str())
            # individual speakers
            for spk in spks:
                fh.writelines(spk.attr_str())

    def load_speaker_table(self, file=None):
        """
        load speaker information from a file and generate a list of speakers
        Args:
            file: str or path, if None use self.speaker_file
        """
        if file is None:
            if self.speaker_file is not None:
                file = self.speaker_file
            elif self.setup is not None:
                file = self._filename_from_setup(self.setup)
                self.speaker_file = file
        # load .txt file using np.loadtxt
        self.speakers = []
        speaker_table = np.loadtxt(file, skiprows=1, delimiter=",", dtype=str)
        for row in speaker_table:
            spk = Speaker()
            for idx, val in enumerate(row):
                if val:
                    setattr(spk, SPEAKER_TABLE_COLS[idx][0],
                            SPEAKER_TABLE_COLS[idx][1](val))
            self.speakers.append(spk)

    def load_calibration(self, file=None, newest=True):
        """
        load calibration results
        :param file: str, calibration file to be loaded
        :param newest: bool, if only loads the newest calibration results
        :return: None
        """
        # TODO
        # important saved calibration results:
        #    SPL_ref: int, id of reference speaker
        #    SPL_const: float, constant used to set slab calibration constant
        #    SPL_eq_spks, list of int, id of speakers in SPL equalization result
        #    SPL_eq: list of float, dB differences w.r.t. reference
        #    filters_spks: list of int, id of speakers in spectrum equalization result
        #    filters: 2d array, list of equalizing filters; 2nd dimension is the different filters
        # saved calib files
        if file is not None:
            self._load_calib_file(file)
        else:
            folder = get_config('CAL_ROOT')
            pattern = 'Calib_{}_*.pkl'.format(self.setup)
            calib_files = []
            for f in os.listdir(folder):
                if fnmatch.fnmatch(f, pattern):
                    calib_files.append(f)
            if not calib_files:
                raise ValueError('no calibration result found')
            if len(calib_files) == 1 or newest:
                calib_dates = self._extract_dates(calib_files)
                idx = calib_dates.index(max(calib_dates))
                file = calib_files[idx]
                self._load_calib_file(file)
            else:
                while True:
                    print('multiple calibration files exist:')
                    print(*calib_files, sep='\n')
                    idx = input("please choose one file to load (using index, starting from 0, "
                                "or 'x' to break): ")
                    try:
                        idx = int(idx)
                    except ValueError:
                        if idx == 'x':
                            print('no calibration result loaded')
                            break
                        print('Please give an integer as the index of chosen file!')
                        continue
                    try:
                        file = calib_files[idx]
                        self.calib_result = self.load_calibration(file)
                        break
                    except IndexError:
                        print('index: {} out of range!'.format(idx))

    @staticmethod
    def _extract_dates(str_list):
        """
        extract dates from list of calibration file names
        :param str_list: list of file names
        :return: list date objects (datetime package)
        """
        dates = []
        for f in str_list:
            match = re.search(r'\d{4}-\d{2}-\d{2}', f)
            dates.append(datetime.datetime.strptime(match.group(), '%Y-%m-%d').date())
        return dates

    def _load_calib_file(self, filename):
        """
        load calibration result from a .pkl file
        :param filename: str, if not absolute, then assume it is in get_config('CAL_ROOT')
        :return:
        """
        if not os.path.isabs(filename):
            folder = get_config('CAL_ROOT')
            filename = os.path.join(folder, filename)
        with open(filename, 'rb') as fh:
            self.calib_result = pickle.load(fh)
        self._apply_calib_result()

    def _apply_calib_result(self, level_only=False):
        """
        apply calibration result into each speaker
        :param level_only: bool, if only SPL level equalization is used
        :return:
        """
        if not self.calib_result:
            raise ValueError('Please load a calibration result first')
        # set slab baseline
        slab.set_calibration_intensity(self.calib_result['SPL_const'])
        for idx, id in enumerate(self.calib_result['SPL_eq_spks']):
            spk = self.pick_speakers(id)[0]
            spk.level = self.calib_result['SPL_eq'][idx]
        if not level_only:
            for idx, id in enumerate(self.calib_result['filters_spks']):
                spk = self.pick_speakers(id)[0]
                spk.filter = self.calib_result['filters'].channel(idx)
                spk.filter_hardware = self.calib_result['filters_hardware'].channel(idx)


if __name__ == '__main__':
    desktop = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
    speaker_file = os.path.join(desktop, 'test_speakers.txt')

    spk_array = SpeakerArray(file=speaker_file)
    spk_array.load_speaker_table()
