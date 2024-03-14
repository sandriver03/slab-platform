import importlib

from labplatform.config import get_config
from .speaker_config import SpeakerArray, load_speaker_config, Speaker
from .utilities import importfile, get_system_delay, prepare_hwfilter_coefs
import slab

import os
import numpy as np
import datetime
import logging
import pickle
logger = logging.getLogger(__name__)


# the device class used should implement:
#   play_and_record method, which plays the self.stimulus and return some ndarray
#   filter_enabled method, which indicated if hardware filter is in use
#   get_filter_ntaps method, which returns the number of taps for the filters
#   load_filters method, which loads filter coefficients into hardware


class SpeakerCalibrator:

    def __init__(self, config):
        self.config = config
        # load configuration parameters
        param = load_speaker_config(self.config)
        self.config_param = param
        # load speaker table
        cal_dir = get_config("CAL_ROOT")
        self.speakerArray = SpeakerArray(file=os.path.join(cal_dir, param['speaker_file']))
        self.speakerArray.load_speaker_table()
        # calibration mode
        self.mode = None
        # calibration parameters
        self.calib_param = {
            'SPL_Vrms': [1.],  # Vrms values used in single speaker calibration
            'stim_type': 'chirp',
            'stim_dur': 0.05,
            'stim_freqbound': [2000, 46000],
            'samplerate': 97656.25,
            'ref_spk_id': 0,
            'spks_to_cal': 'all',
            'N_repeats': 10,
            'calib_dB': 75,
            'filter_bank': {'length': 512,
                            'bandwidth': 0.125,
                            'low_cutoff': 2000,
                            'high_cutoff': None,
                            'alpha': 1.0},
            'stim_ramp': {'duration': 0.005, }    # parameters for stimulus ramp, see slab.Sound.ramp
        }
        if 'ref_spk_id' in self.config_param:
            self.calib_param['ref_spk_id'] = self.config_param['ref_spk_id']
        else:
            # by default, use the 1st speaker as reference
            self.calib_param['ref_spk_id'] = 0
        # sound wave data
        self.audio_data = None
        self.save_audio_data = False
        self.stimulus = None
        # custom stimulus should be generated using slab toolbox
        self.generate_default_stim()
        # calibration results
        self.results = dict()
        # file path
        self._path = cal_dir
        self._datestr = datetime.datetime.now().strftime(get_config("DATE_FMT"))
        # initialize the device used to perform speaker calibration
        # the device should be an instance of Device class
        py_file = param['device']['file_name']
        if not os.path.isabs(py_file):
            device_dir = get_config("DEVICE_ROOT")
            py_file_full = os.path.join(device_dir, py_file)
        else:
            py_file_full = py_file
        if os.path.isfile(py_file_full):
            device_mod = importfile(py_file_full)
        else:
            device_mod = importlib.import_module(py_file)
        if not device_mod:
            raise RuntimeError('Device class cannot be loaded programmatically. Please load it manually')
        self.device = getattr(device_mod, param['device']['device_class'])()
        # time.sleep(0.5)
        self.change_mode('SPL')

    def generate_default_stim(self):
        """
        generate stimulus used to perform SPL/spectrum equalization
        :return:
        """
        stim_class = getattr(slab.Sound, self.calib_param['stim_type'])
        self.stimulus = stim_class(duration=self.calib_param['stim_dur'],
                                   from_frequency=self.calib_param['stim_freqbound'][0],
                                   to_frequency=self.calib_param['stim_freqbound'][1],
                                   samplerate=self.calib_param['samplerate'])

    def change_mode(self, new_mode):
        """
        change calibration mode
        :param new_mode: str, currently 'SPL', 'SPL_eq', 'spec_eq', 'test', 'test_hw'
        :return:
        """
        if new_mode != self.mode:
            if new_mode == 'SPL':
                logger.info('calibrating the sound pressure level of a single speaker')
                # default settings
                logger.info('by default, using white noise. modify the settings through the device if needed')
                self.device.configure(use_tone=0, use_noise=1, use_custom=0, use_pulse=0,
                                      WN_amp=1, filter_on=False)
            elif new_mode == 'SPL_eq':
                logger.info('equalizing the sound pressure level of different speakers w.r.t a reference speaker')
                # default settings
                logger.info('by default, using a 50ms chirp sound as stimulus. modify the settings through the device '
                            'if needed')
                self.device.configure(use_tone=0, use_noise=0, use_custom=1, use_pulse=1, filter_on=False)
            elif new_mode == 'spec_eq':
                logger.info('measuring the filter sets that can be used to remove spectral differences across different'
                            ' speakers')
                # default settings
                logger.info('by default, using a 50ms chirp sound as stimulus. modify the settings through the device '
                            'if needed')
                self.device.configure(use_tone=0, use_noise=0, use_custom=1, use_pulse=1, filter_on=False)
            elif new_mode == 'test':
                logger.info('testing the speaker equalization. by default, test the spectrum equalization. can also '
                            'only test the SPL equalization')
                # default settings
                logger.info('by default, using a 50ms chirp sound as stimulus. modify the settings through the device '
                            'if needed')
                self.device.configure(use_tone=0, use_noise=0, use_custom=1, use_pulse=1, filter_on=False)
            elif new_mode == 'test_hw':
                logger.info('testing the speaker equalization using the hardware filters')
                # default settings
                logger.info('by default, using a 50ms chirp sound as stimulus. modify the settings through the device '
                            'if needed')
                self.device.configure(use_tone=0, use_noise=0, use_custom=1, use_pulse=1, filter_on=True)
            else:
                raise ValueError('mode: {} is not implemented'.format(new_mode))

            self.mode = new_mode

    def calibrate(self, target_spks=None, ref_spk=None):
        """
        calibrate the speaker Array
        first requires the user to use an SPL meter to measure the SPL from the reference speaker, then performs:
            1. equalizing the SPL of other speakers w.r.t. the reference speaker
            2. equalizing the frequency response of other speakers w.r.t. the reference speaker
        :param target_spks: list of Speaker instances (returned from SpeakerArray.pick_speakers)
        :param ref_spk: a Speaker instance, reference speaker
        :return: None
        """
        self.change_mode('SPL')
        # measure reference speaker
        self._calibrate_SPL(ref_spk)
        # SPL equalization levels
        self.change_mode('SPL_eq')
        _ = self._equalize_SPL(target_spks, ref_spk)
        # spectrum equalization filters
        self.change_mode('spec_eq')
        _ = self._equalize_spectrum(target_spks, ref_spk)
        # save results
        self._save_result()

    def _save_result(self):
        """
        save complete calibration result
        :return:
        """
        # file name should be Calib_{config_name}_{date_str}.pkl
        fname = 'Calib_{}_{}.pkl'.format(self.config, self._datestr)
        fpath = os.path.join(self._path, fname)
        ref_spl_str = "SPL_{}_{}".format(self.calib_param['ref_spk_id'], self.config)
        # TODO: overriding?
        if os.path.isfile(fpath):
            logger.warning('calibration file "{}" already exists. overriding old result...'.format(fname))
        with open(fpath, 'wb') as fh:
            pickle.dump(self.results, fh, protocol=pickle.HIGHEST_PROTOCOL)

    def _calibrate_SPL(self, spk=None, save=False):
        """
        calibrate single speaker SPL
        play a constant sound from a speaker, and use SPL meter to read the dB level
        currently using 4 different levels and output. results are saved as (Vrms, dB)
        :param spk: int, or instance of Speaker
        :param save: bool, if save the result to a separate file
        :return: None
        """
        if self.mode != 'SPL':
            raise RuntimeError('Please first set the mode to "SPL"')

        # check if a Speaker instance is provided
        if spk is None:
            spk = self.calib_param['ref_spk_id']
        if isinstance(spk, int):
            spk = self.speakerArray.pick_speakers(spk)[0]
        if not isinstance(spk, Speaker):
            raise ValueError('input argument spk must be an instance of Speaker class, or an integer')
        self.calib_param['ref_spk_id'] = spk.id
        # check if using pure tone or white noise
        if self.device.setting.use_noise:
            use_noise = 1
        else:
            use_noise = 0
        # set analog output channel to chosen speaker
        # TODO: for now it is easy since we only have one output device. for multiple output devices, we need a mapping
        #  from speaker attributes to output device
        self.device.configure(stim_ch=spk.channel_analog)
        SPL_res = []
        for amp in self.calib_param['SPL_Vrms']:
            if use_noise:
                self.device.configure(WN_amp=amp)
            else:
                self.device.configure(tone_amp=amp)
            # play continuous sound from the chosen speaker
            print('Playing sound from the speaker: {}. Please measure intensity.'.format(spk.id))
            self.device.start()
            intensity = float(input('Enter measured intensity in dB: '))  # ask for measured intensity
            SPL_res.append((amp, intensity))
            # stop the sound
            self.device.pause()

        # dB/Vrms ratio at Vrms = 1; used as a constant
        if 1 in self.calib_param['SPL_Vrms']:
            idx = self.calib_param['SPL_Vrms'].index(1)
            intensity_const = SPL_res[idx][1] - 20.0*np.log10(1/2e-5)
        else:
            intensity_const = None
        self.results['SPL_ref'] = spk.id
        self.results['SPL_const'] = intensity_const
        self.results['SPL_res'] = SPL_res
        self.results['calib_param'] = self.calib_param

        if save:
            # generate file name
            spl_fname = "SPL_{}_{}_{}.pkl".format(spk.id, self.config, self._datestr)
            # save result as pickle file
            abs_fname = os.path.join(self._path, spl_fname)
            if os.path.exists(abs_fname):
                print("file {} already exists. Overriding it...".format(abs_fname))
            with open(abs_fname, 'wb') as fh:
                pickle.dump({'const': intensity_const, 'res': SPL_res}, fh,
                            protocol=pickle.HIGHEST_PROTOCOL)

    def _equalize_SPL(self, target_spks=None, ref_spk=None, save=False):
        """
        Record the signal from each speaker in the list and return the level of each speaker relative to the reference
        speaker
        :param target_spks: list of Speaker instances (returned from SpeakerArray.pick_speakers)
        :param ref_spk: a Speaker instance, reference speaker
        :param save: bool, if save the result to a separate file
        :return: ndArray
        """
        if self.mode != 'SPL_eq':
            raise RuntimeError('Please first set the mode to "SPL_eq"')

        if ref_spk is None:
            ref_spk = self.speakerArray.pick_speakers(self.calib_param['ref_spk_id'])[0]
        else:
            ref_spk = self.speakerArray.pick_speakers(ref_spk)
        if target_spks is None:
            target_spks = self.speakerArray.pick_speakers(self.calib_param['spks_to_cal'])
        else:
            target_spks = self.speakerArray.pick_speakers(target_spks)

        # use calibrated ref_spk to set the correct stimulus strength
        # TODO: load single speaker SPL measurement if _calibrate_SPL is not done
        if 'SPL_const' in self.results and ref_spk.id == self.results['SPL_ref']:
            slab.set_calibration_intensity(self.results['SPL_const'])
        else:
            raise RuntimeError('SPL for the reference speaker need to be measured first')
        # set stimulus intensity
        self.stimulus.level = self.calib_param['calib_dB']

        N_reps = self.calib_param['N_repeats']
        result = np.zeros((len(target_spks), N_reps))
        da_diff = np.zeros((1, N_reps))    # dB differences at digital (displayed in slab) and analog (recorded) level
        # actual stimulus is ramped
        stim_ramped = self.stimulus.ramp(**self.calib_param['stim_ramp'])
        for n in range(N_reps):
            ref_rec = self._play_record_custom(self.stimulus, ref_spk)
            ref_rec = slab.Sound(ref_rec, samplerate=self.calib_param['samplerate'])
            recordings = []
            for spk in target_spks:
                recordings.append(self._play_record_custom(self.stimulus, spk))
            recordings = slab.Sound(recordings, samplerate=self.calib_param['samplerate'])
            result[:, n] = recordings.level - ref_rec.level
            da_diff[0, n] = ref_rec.level - stim_ramped.level

        self.results['SPL_eq'] = result.mean(1)
        self.results['SPL_eq_raw'] = result
        self.results['SPL_eq_spks'] = [spk.id for spk in target_spks]
        self.results['SPL_eq_ref'] = ref_spk.id
        self.results['SPL_eq_dadiff'] = da_diff.mean()

        if save:
            # generate file name
            spl_fname = "SPLeq_{}_{}.pkl".format(self.config, self._datestr)
            # save result as pickle file
            abs_fname = os.path.join(self._path, spl_fname)
            if os.path.exists(abs_fname):
                print("file {} already exists. Overriding it...".format(abs_fname))
            with open(abs_fname, 'wb') as fh:
                pickle.dump({'levels': result.mean(1),
                             'spk_info': self.results['SPL_eq_spks'],
                             'raw': result},
                            fh, protocol=pickle.HIGHEST_PROTOCOL)

        return result

    def _play_record_custom(self, stim, spk, baseline_correct=True):
        """
        load a custom stimulus, and play it from a speaker and record the result
        :param stim: slab.Sound instance
        :param spk: Speaker instance
        :param baseline_correct: Bool, if set baseline to 0
        :return: np array
        """
        # load stimulus into the play buffer
        self.device.load_stimulus(stimulus=stim)
        # record from speaker
        spk_delay = self._get_speaker_delay(spk)
        if self.device.filter_enabled():
            rec_delay = spk_delay + int(self.device.get_filter_ntaps()/2)
            gate_delay = int(self.device.get_filter_ntaps()/2)
        else:
            rec_delay = spk_delay
            gate_delay = 1
        self.device.configure(stim_ch=spk.channel_analog,
                              sys_delay_n=rec_delay,
                              gate_delay_n=gate_delay,
                              stim_length=stim.duration * 1000)
        ref_rec = self.play_and_record()
        if baseline_correct:
            ref_rec = ref_rec - ref_rec.mean()
        return ref_rec

    def _equalize_spectrum(self, target_spks=None, ref_spk=None, save=False):
        """
        calculate filter bank that can be used to equalize the spectrum of each speaker w.r.t. a reference speaker
        :param target_spks: list of Speaker instances (returned from SpeakerArray.pick_speakers)
        :param ref_spk: a Speaker instance, reference speaker
        :param save: bool, if save the result to a separate file
        :return:
        """
        if self.mode != 'spec_eq':
            raise RuntimeError('Please first set the mode to "spec_eq"')

        if ref_spk is None:
            ref_spk = self.speakerArray.pick_speakers(self.calib_param['ref_spk_id'])[0]
        else:
            ref_spk = self.speakerArray.pick_speakers(ref_spk)
        if target_spks is None:
            target_spks = self.speakerArray.pick_speakers(self.calib_param['spks_to_cal'])
        else:
            target_spks = self.speakerArray.pick_speakers(target_spks)

        # should use only one round of recording
        recordings = []
        for spk in target_spks:
            rec = self._play_record_custom(self.stimulus, spk)
            recordings.append(rec - rec.mean())
            # recordings.append(self._play_record_custom(self.stimulus, spk))
        recordings = slab.Sound(recordings, samplerate=self.calib_param['samplerate'])
        # use slab to calculate filter banks
        # TODO: should we consider the baseline shift (calibration intensity) here?
        # calculate both software (filtfilt) and hardware (causal) filters
        # we should aim to reduce the difference between digital and recorded signal
        digit_stim = self.stimulus.ramp(**self.calib_param['stim_ramp'])
        digit_stim.level += self.results['SPL_eq_dadiff']
        filter_bank = slab.Filter.equalizing_filterbank(digit_stim, recordings, **self.calib_param['filter_bank'])
        self.results['filters'] = filter_bank
        filter_bank = slab.Filter.equalizing_filterbank(digit_stim, recordings, **self.calib_param['filter_bank'],
                                                        filt_meth='causal')
        self.results['filters_hardware'] = filter_bank
        self.results['filters_spks'] = [spk.id for spk in target_spks]
        self.results['filters_ref'] = ref_spk.id

        if save:
            # generate file name
            spl_fname = "SPLeq_{}_{}.pkl".format(self.config, self._datestr)
            # save result as pickle file
            abs_fname = os.path.join(self._path, spl_fname)
            if os.path.exists(abs_fname):
                print("file {} already exists. Overriding it...".format(abs_fname))
            with open(abs_fname, 'wb') as fh:
                pickle.dump({'filters': filter_bank, 'spk_info': self.results['filters_spks']}, fh,
                            protocol=pickle.HIGHEST_PROTOCOL)

        return filter_bank

    def play_and_record(self):
        """
        play pre-loaded sound from the device and make the recording.
        :return: ndArray
        """
        # TODO: the device class need to implement the play_and_record method
        return self.device.play_and_record().flatten()

    def _get_speaker_delay(self, spk, temp=20.):
        """
        get system delay in n samples for a given speaker
        :param spk: a Speaker instance
        :param temp: float, temperature in celsius
        :return: int
        """
        return get_system_delay(spk.distance, temp, self.calib_param['samplerate'],
                                self.config_param['TDT_aud'],
                                self.config_param['TDT_rec'])

    def stop(self):
        """
        stop the hardware
        """
        self.device.stop()

    def _set_SPL_level(self, SPL_val, spk=None):
        """
        manually set an SPL level for a speaker. used for debugging
        :param SPL_val: float, dB level of the speaker when TDT output amplitude is 1V
        :param spk: a Speaker instance
        """
        if spk is None:
            spk = self.speakerArray.pick_speakers(self.calib_param['ref_spk_id'])[0]
        else:
            spk = self.speakerArray.pick_speakers(spk)

        intensity_const = SPL_val - 20.0 * np.log10(1 / 2e-5)
        self.results['SPL_const'] = intensity_const
        self.results['SPL_ref'] = spk.id

    def test_equalization(self, stim=None, spks=None, level_only=False, raw=False):
        """
        test the equalization of different speakers, using software equalization
        :param stim: slab.Sound
        :param spks: list-like, speakers to be tested
        :param level_only: bool, if only level equalization is tested
        :param raw: bool, if true get outputs without any modification
        :return:
        """
        if self.mode not in ('test', 'test_hw'):
            raise RuntimeError('Please first set the mode to "test" or "test_hw"')

        # check if calibration is already done
        if not self.speakerArray.calib_result:
            self.speakerArray.calib_result = self.results
            self.speakerArray._apply_calib_result()
        if stim is None:
            stim = self.stimulus
        if spks is None:
            spks = self.speakerArray.pick_speakers('all')
        else:
            spks = self.speakerArray.pick_speakers(spks)

        if not stim:
            raise ValueError('Please provide a stimulus to use ')
        if not isinstance(stim, slab.Sound):
            stim = slab.Sound(stim, samplerate=self.calib_param['samplerate'])
        # set stimulus level to calibration level
        stim.level = self.calib_param['calib_dB']

        # load filter coefficients
        if self.mode == 'test_hw':
            coefs, ntaps = prepare_hwfilter_coefs(spks)
            self.device.load_filters(coefs, ntaps)
            if raw:
                # disable filter
                self.device.configure(filter_on=False)

        res = []
        for spk in spks:
            # apply software equalization
            if self.mode == 'test':
                if not raw:
                    stim_final = spk.apply_equalization(stim, level_only=level_only)
                else:
                    stim_final = stim
            else:
                if level_only:
                    raise RuntimeError('for hardware filters, level only equalization is not implemented')
                # select corresponding hardware filter
                self.device.configure(filter_select=spk.id)
                stim_final = stim
            res.append(self._play_record_custom(stim_final, spk))
        return slab.Sound(res, samplerate=self.calib_param['samplerate'])


if __name__ == '__main__':
    import logging
    log = logging.getLogger()
    log.setLevel(logging.WARNING)
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    log.addHandler(ch)

    spk_cal = SpeakerCalibrator('IMAGING')
    # fd.setting.configure_traits()
