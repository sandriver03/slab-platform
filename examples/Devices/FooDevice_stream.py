'''
    test device, random number generator
'''

from labplatform.core.Setting import DeviceSetting
from labplatform.core.Device import Device

import numpy as np
import queue
from traits.api import Int, Instance, Float, Any, Property, Str, CInt, Tuple, Bool
# from PyQt4.QtCore import QTimer

import logging
log = logging.getLogger(__name__)


class FooDeviceSetting(DeviceSetting):
    sampling_freq = CInt(1000, group='primary', dsec='number of data points per channel per second. (Hz)')
    shape         = Tuple((-1, 16), group='primary', dsec='number of channels; each channel generates one stream '
                                                          'of random numbers')
    buffer_size   = CInt(10, group='primary', dsec='buffer size; numbers of data chunk. One control interval '
                                                   'generates one data chunk')
    mu            = Float(1, group='primary', dsec='mean of the random numbers', reinit=False)
    sigma         = Float(1, group='primary', dsec='std of the random numbers', reinit=False)
    data_length   = CInt(-1, group='primary', dsec='if positive, the device will stop after reaching the '
                                                   'desired length. otherwise it runs indefinitely')
    dtype = Str('float', group='primary', dsec='data type of the device. device property')
    software_scaling = Int(2, group='primary', dsec='software scaling factor')

    buffer_length = Property(CInt, group='derived', depends_on=['sampling_freq', 'buffer_size',
                                                                'control_interval', 'data_length'],
                             dsec='length of internal buffer required')
    chunk_size = Property(Int, group='derived', depends_on=['sampling_freq', 'control_interval'],
                          dsec='number of data point generated per timer interval')

    device_type = 'FooDevice'

    def _get_buffer_length(self):
        if self.data_length <= 0:
            return int(self.control_interval * self.sampling_freq * self.buffer_size)
        else:
            return self.data_length

    def _get_chunk_size(self):
        return int(self.control_interval * self.sampling_freq)

    '''
    def __global_paras_lookup_default(self):
        return {'control_interval': 'CONTROL_INTERVAL',
                            'max_analogout_voltage': 'MAX_ANALOGOUT_VOLTAGE',
                            'noise_duration': 'NOISE_DURATION'}
    '''


# parameters require to reset stream
stream_params = {'shape', }


class FooDevice(Device):
    """
    generate streams of Gaussian random numbers when run
    """
    buffer        = Any(dsec='internal buffer, must have `buffer.shape[0] == nb_channel`.')
    buffer_widx   = Int(0, dsec='write index of the buffer')
    buffer_ridx   = Int(0, dsec='read index of the buffer')
    n_channel     = Int(0, dsec='number of data channels')
    chunk_size    = Int(0)
    buffer_length = Int(0)

    setting = Instance(FooDeviceSetting, ())

    # here we want to use the default thread
    # _use_default_thread = True
    _use_default_thread = True

    # if stream need to be reset
    _stream_para_changed = Bool(False)
    _empty_data = Any    # used to send trial stop signal
    _packet_N = 0   # debug

    def _initialize(self, **kwargs):
        # setup stream
        if set(self._changed_params) & stream_params:
            self._stream_para_changed = True
        # setting up output stream
        if self.output and self._stream_para_changed:
            # close and delete old stream
            names = list(self.output.keys())
            for out_name in names:
                self.remove_stream(out_name, 'output')
            self.output = None
        if not self.output:
            # create output stream
            self.create_output_stream('tcp', monitor=True)

        self.buffer_ridx, self.buffer_widx = 0, 0
        # prepare internal buffer
        self.buffer = np.empty([int(self.setting.buffer_length), self.setting.shape[1]],
                               dtype=self.setting.dtype)
        # prepare empty data
        self._empty_data = np.zeros((1, ) + self.setting.shape[1:], self.setting.dtype)

    def configure(self, **kwargs):
        """
        Parameters
        ----------
        shape: tuple, (0, 16)
            Number of output channels.
        sampling_freq: float, 1000
            Number of data points generated per second
        buffer_size: int, 10
            Data to send. numbers of data chunk. One control interval generates one data chunk.
        mu: float, 1
            Average of the random numbers
        sigma: float, 1
            std of the random numbers
        control_interval: int, 0.5
            interval of internal timer, second
        data_length: int, -1
            if positive, the device will stop after reaching the desired length. otherwise it runs
            indefinitely; when it is set, buffer_size and control_interval has no use
        """
        return Device.configure(self, **kwargs)

    def _start(self):
        pass

    def _configure(self, **kwargs):
        self.n_channel = self.setting.shape[1]
        self.chunk_size = self.setting.chunk_size
        self.buffer_length = self.setting.buffer_length

    def _pause(self):
        pass

    def _stop(self):
        pass

    def _deinitialize(self):
        fd.buffer = None

    def thread_func(self):
        # write data into self buffer
        self.buffer_widx = self.buffer_widx % self.setting.buffer_length
        # data = self.setting.mu + self.setting.sigma * \
        #            np.random.randn(self.chunk_size, self.n_channel)
        data = self._packet_N * np.ones((self.chunk_size, self.n_channel), dtype=self.setting.dtype)
        self._packet_N += 1
        self.buffer[self.buffer_widx:(self.buffer_widx+self.chunk_size), :] = data
        self.buffer_widx += self.chunk_size

        # send data through stream; mark data with header of b'd'
        self.output[self.name + '_tcp'].send(data, header=(b'd', ))

        if self.setting.data_length > 0:
            self.stop_at_datalength()

    def generate_data_on_timer(self):
        # generate data and put it into buffer
        self.buffer_widx = self.buffer_widx % self.buffer_length
        data = self.setting.mu + self.setting.sigma * np.random.randn(self.chunk_size, self.n_channel)
        self.buffer[self.buffer_widx:(self.buffer_widx + self.chunk_size), :] = data
        self.buffer_widx += self.chunk_size

    def stop_at_datalength(self):
        if self.buffer_widx >= self.setting.data_length:
            self.pause()
            # send a trial stop signal to the stream
            self.output[self.name + '_tcp'].send(self._empty_data, header=(b's',))
            if self.experiment:
                self.experiment().process_event({'trial_stop': 0})


# test
if __name__ == '__main__':
    log = logging.getLogger()
    log.setLevel(logging.DEBUG)
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    log.addHandler(ch)

    fd = FooDevice()
    # fd.setting.configure_traits()
