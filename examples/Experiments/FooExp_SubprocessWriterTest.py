"""
    a testing case, operation on the FooDevice_stream
"""
from labplatform.config import get_config
from labplatform.core.ExperimentLogic import ExperimentLogic
from labplatform.core.Setting import ExperimentSetting
from labplatform.core.Data import ExperimentData
from labplatform.utilities.arraytools import downscale_nparray
from examples.Devices.FooDevice_stream import FooDevice

from traits.api import List, Instance, Float, Property, Int
import random
import numpy as np
import time

import logging
log = logging.getLogger()


class FooExperimentSetting(ExperimentSetting):
    experiment_name = 'FooExp'
    mu_sequence = List([1, 1], group='primary', context=False, dsec='different means of the Gaussian to run')
    trial_duration = 5
    trial_number = 5

    total_trial = Property(Int(), group='status', depends_on=['trial_number', 'mu_sequence'],
                                    dsec='Total number of trials')

    def _get_total_trial(self):
        return self.trial_number * len(self.mu_sequence)


class FooExperiment(ExperimentLogic):
    setting = FooExperimentSetting()
    data = ExperimentData()
    time_0 = Float()

    def _devices_default(self):
        fd = FooDevice()
        fd.setting.device_ID = 0
        return {'FooDevice': fd}

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
        data_length = self.devices['FooDevice'].setting.sampling_freq * self.setting.trial_duration
        self.devices['FooDevice'].configure(data_length=data_length)
        self.time_0 = time.time()
        # set data class to use thread writer
        self.data.writer_type = 'subprocess'
        # put stream handler in writer_params
        self.data.configure_writer(stream_handler={'FooDevice_0_FooDevice_0_tcp': foo_stream_handler})

    def _before_start(self):
        # register stream handler
        # self.data.writer.worker.register_stream_handler('FooDevice_0_FooDevice_0_tcp', foo_stream_handler)
        pass

    def _start_trial(self):
        self.devices['FooDevice'].configure(mu=self.mu_list[self.setting.current_trial])
        self.devices['FooDevice'].start()
        log.info('trial {} start: {}'.format(self.setting.current_trial, time.time() - self.time_0))
        self.time_0 = time.time()

    def _stop_trial(self):
        # read data from FooDevice
        # self.data.write('FooDevice_0', self.devices['FooDevice'].buffer)
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


# TODO: problem: in python, it seems the objects defined in __main__ cannot be pickled
# it probably make sense in this case to define handlers in a separate file
def foo_stream_handler(obj, stream, stop_sig=b's', data_sig=b'd', preview_sig=b'p',
                       pkt_rt=10, pkt_size=100):
    """
    read camera stream data
    Args:
        obj: the writer.worker instance
        stream: an Stream.InputStream object
        stop_sig: byte, stop signal from the stream
        data_sig: byte, stream packets need to be saved
        preview_sig: byte, stream packets need to be skipped
        pkt_rt: number of packet per control interval
        pkt_size: length of each packet
    Returns:
    """
    # first check how many packets are queued up
    try:
        packet_to_read = stream.monitor()[0] - stream.N_packet
        # controlling ran away RAM usage
        if packet_to_read > 10 * pkt_rt:
            packet_to_read = 10 * pkt_rt
    except TypeError:  # stream.monitor returns None because no new data arrived
        if stream.latest_packet_N is not None:
            packet_to_read = stream.latest_packet_N - stream.N_packet
        else:
            packet_to_read = stream.Idx_initial_packet - stream.N_packet
    if packet_to_read <= 0:
        return
    flag_idx = obj.input_names.index(stream.name)
    if not obj._input_finished_flags[flag_idx]:
        # create array
        data = np.zeros(((packet_to_read * pkt_size, ) + stream.params['shape'][1:]),
                        dtype=stream.params['dtype'])
        # read queued packets and save it to file
        # only do this when there is actual data to save, i.e. camera is not in preview mode
        tosave_idx = 0
        for idx in range(packet_to_read):
            s_data = stream.recv()
            if s_data[2][1] == stop_sig:
                data = data[:tosave_idx]
                obj.input_finished(obj._current_trial, (stream.name,))
                break
            elif s_data[2][1] == data_sig:
                data[idx*pkt_size:(idx+1)*pkt_size] = s_data[1]
                tosave_idx += 1 * pkt_size
        if tosave_idx > 0:
            # down scale array if software scaling is specified
            if 'software_scaling' in stream.params and stream.params['software_scaling']:
                data = downscale_nparray(data, (1,) + (stream.params['software_scaling'],))
            obj.write(stream.name, data, obj._current_trial)


if __name__ == '__main__':
    from labplatform.core.Subject import load_cohort
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

    sl = load_cohort('example')

    fe = FooExperiment(subject=sl.subjects[1])
    fd = fe.devices['FooDevice']
    # fe.start()
