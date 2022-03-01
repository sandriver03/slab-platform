'''
    a testing case, operation on the FooDevice
'''

from labplatform.config import get_config
from labplatform.core.ExperimentLogic import ExperimentLogic
from labplatform.core.Setting import ExperimentSetting
from labplatform.core.Data import ExperimentData

from examples.Devices.ShamCam import shamCam

from traits.api import List, Instance, Float, Property, Int
import random
import numpy as np
import time

import logging
log = logging.getLogger(__name__)


class FooCamExperimentSetting(ExperimentSetting):
    experiment_name = 'FooCamExp'
    mu_sequence = List([1, 1], group='primary', context=False, dsec='different means of the Gaussian to run')
    trial_duration = 5
    trial_number = 5
    test_large_val = Instance(np.ndarray, group='primary', context=False, dsec='test large val')

    total_trial = Property(Int(), group='status', depends_on=['trial_number'],
                                    dsec='Total number of trials')

    def _test_large_val_default(self):
        return np.random.randn(2, 1)

    def _get_total_trial(self):
        return self.trial_number


class FooCamExperiment(ExperimentLogic):
    setting = FooCamExperimentSetting()
    data = ExperimentData()
    time_0 = Float()

    def _devices_default(self):
        fd = shamCam()
        fd.setting.device_ID = 0
        return {'Cam': fd}

    def _initialize(self, **kwargs):
        pass

    # internal temporal parameter
    mu_list = List()

    def setup_experiment(self, info=None):
        self.mu_list = []
        for k in self.setting.mu_sequence:
            self.mu_list.extend([k]*self.setting.trial_number)
        # randomize the sequence
        random.shuffle(self.mu_list)
        # save the sequence
        self._tosave_para['mu_sequence'] = self.mu_list

        # setup correct data_length in FooDevice
        data_length = self.devices['Cam'].setting.sampling_freq * self.setting.trial_duration
        self.devices['Cam'].configure(data_length=data_length)
        self.time_0 = time.time()

    def _before_start(self):
        self.data.close_input('ShamCam_0_ShamCam_0_tcp')

    def _start_trial(self):
        # self.devices['Cam'].configure(mu=self.mu_list[self.setting.current_trial])
        # reset cam buffer index
        self.devices['Cam'].buffer.reset_index()
        self.devices['Cam'].start()
        log.info('trial {} start: {}'.format(self.setting.current_trial, time.time() - self.time_0))
        self.time_0 = time.time()

    def _stop_trial(self):
        # read data from FooDevice
        self.data.write('ShamCam_0_ShamCam_0_tcp', self.devices['Cam'].buffer[:])
        # self.data.current['FooDevice_0'].write(self.devices['FooDevice'].buffer)
        # save data
        self.data.save()
        log.info('trial {} end: {}'.format(self.setting.current_trial, time.time() - self.time_0))
        self.time_0 = time.time()

    def _stop(self):
        pass

    def _pause(self):
        pass


if __name__ == '__main__':
    from labplatform.core.Subject import SubjectList
    from labplatform.config import get_config
    import os
    import logging

    log = logging.getLogger()
    log.setLevel(logging.INFO)
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    log.addHandler(ch)

    h5path = os.path.join(get_config('SUBJECT_ROOT'), 'Test_0.h5')
    sl = SubjectList(file_path=h5path)
    sl.read_from_h5file()

    fe = FooCamExperiment(subject=sl.subjects[0])
    fd = fe.devices['Cam']
    # fe.start()
