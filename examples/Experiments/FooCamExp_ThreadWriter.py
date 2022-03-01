'''
    a testing case, operation on the FooDevice
'''

from labplatform.config import get_config
from labplatform.core.ExperimentLogic import ExperimentLogic
from labplatform.core.Setting import ExperimentSetting
from labplatform.core.Data import ExperimentData
from labplatform.utilities.arraytools import downscale_nparray

from examples.Devices.ShamCam import shamCam

from traits.api import List, Instance, Float, Property, Int
import random
import numpy as np
import time

import logging
log = logging.getLogger(__name__)


class FooCamExperimentSetting(ExperimentSetting):
    experiment_name = 'FooCamExp'
    mu_sequence = List([1, 2], group='primary', context=False, dsec='different means of the Gaussian to run')
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
        cam = shamCam()
        cam.setting.device_ID = 0
        return {'Cam': cam}

    def _initialize(self, **kwargs):
        self.devices['Cam'].configure(buffer_time=2, )

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
        # set data class to use thread writer
        self.data.writer_type = 'thread'
        self.data.configure_writer(stream_handler={'ShamCam_0_ShamCam_0_tcp': cam_stream_handler})

    def _before_start(self):
        # self.data.close_input('ShamCam_0_ShamCam_0_tcp')
        # self.data.writer.worker.register_stream_handler('ShamCam_0_ShamCam_0_tcp', cam_stream_handler)
        pass

    def _start_trial(self):
        # self.devices['Cam'].configure(mu=self.mu_list[self.setting.current_trial])
        # reset cam buffer index
        self.devices['Cam'].buffer.reset_index()
        self.devices['Cam'].start()
        log.info('trial {} start: {}'.format(self.setting.current_trial, time.time() - self.time_0))
        self.time_0 = time.time()

    def _stop_trial(self):
        # read data from FooDevice
        # self.data.write('ShamCam_0_ShamCam_0_tcp', self.devices['Cam'].buffer[:])
        # self.data.current['FooDevice_0'].write(self.devices['FooDevice'].buffer)
        # set trial end flags
        self.data.input_finished(('event_log', 'trial_log'))
        # save data
        self.data.save(close_file=False)
        log.info('trial {} end: {}'.format(self.setting.current_trial, time.time() - self.time_0))
        self.time_0 = time.time()

    def _stop(self):
        pass

    def _pause(self):
        pass


def cam_stream_handler(writer_obj, stream, stop_sig=b's'):
    """
    read camera stream data
    Args:
        writer_obj: the writer.worker instance
        stream: an Stream.InputStream object
        stop_sig: byte, stop signal from the stream
    Returns:
    """
    # first check how many packets are queued up
    try:
        packet_to_read = stream.monitor()[0] - stream.N_packet
    except TypeError:  # stream.monitor returns None because no new data arrived
        if stream.latest_packet_N is not None:
            packet_to_read = stream.latest_packet_N - stream.N_packet
        else:
            packet_to_read = stream.Idx_initial_packet - stream.N_packet
    if packet_to_read <= 0:
        return
    flag_idx = writer_obj.input_names.index(stream.name)
    if not writer_obj._input_finished_flags[flag_idx]:
        # create array
        data = np.zeros(((packet_to_read, ) + stream.params['shape'][1:]),
                        dtype=stream.params['dtype'])
        # read queued packets and save it to file
        for idx in range(packet_to_read):
            s_data = stream.recv()
            if s_data[2][1] == stop_sig:
                data = data[:idx]
                writer_obj.input_finished(writer_obj._current_trial, (stream.name, ))
                break
            else:
                data[idx] = s_data[1]
        # down scale array if software scaling is specified
        if 'software_scaling' in stream.params and stream.params['software_scaling']:
            data = downscale_nparray(data, (1,) + stream.params['sf_full'])
        writer_obj.write(stream.name, data, writer_obj._current_trial)


def cam_stream_handler_v1(object, stream, stop_sig=b's'):
    """
    read camera stream data
    Args:
        object: the writer.worker instance
        stream: an Stream.InputStream object
        stop_sig: byte, stop signal from the stream
    Returns:
    """
    # first check how many packets are queued up
    N_read_max = 7  # maximum number of frames to read in each iteration
    N = 0
    flag_idx = object.input_names.index(stream.name)
    if not object._input_finished_flags[flag_idx]:
        # create array
        data = np.zeros(((N_read_max, ) + stream.params['shape'][1:]),
                        dtype=stream.params['dtype'])
        # read queued packets and save it to file
        for idx in range(N_read_max):
            if stream.poll():
                s_data = stream.recv()
                if s_data[2][1] == stop_sig:
                    data = data[:idx]
                    object.input_finished(object._current_trial, (stream.name, ))
                    break
                else:
                    data[idx] = s_data[1]
            else:
                data = data[:idx]
                break

        if idx > 0:
            object.write(stream.name, data, object._current_trial)


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
