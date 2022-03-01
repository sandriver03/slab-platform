'''
    test device, random number generator
'''

from labplatform.core.Setting import DeviceSetting
from labplatform.core.Device import Device

import numpy as np
# import queue
from traits.api import Int, Instance, Float, Any, Property, Str, CInt, Tuple
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


class FooDevice(Device):
    """
    generate streams of Gaussian random numbers when run
    """

    # timer  = Instance(QTimer)  # timer for data generation ! not in use at the moment
    _output_specs = {'type': 'analog_signal', 'shape': (-1, 16), 'sampling_freq': 1000,
                     'dtype': float, 'length': 0}

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

    def _initialize(self, **kwargs):
        # set up timer
        # self.configure_timer(**kwargs)
        self.buffer_ridx, self.buffer_widx = 0, 0
        # prepare internal buffer
        self.buffer = np.empty([int(self.setting.buffer_length), self.setting.shape[1]],
                               dtype=self._output_specs['dtype'])
        # save output specs
        self._output_specs['shape'] = self.setting.shape
        self._output_specs['sampling_freq'] = self.setting.sampling_freq

    '''
    QTimer requires a Qt event loop to operate. for now, let's use plain python for simplicity
    '''
    '''
    def configure_timer(self, **kwargs):
        if not self.timer:
            self.timer = QTimer()
            print('adding call back function to timer')
            self.timer.timeout.connect(self.generate_data)
        if 'control_interval' in kwargs:
            self.setting.control_interval = kwargs['control_interval']
            log.debug('set attribute {} to value {}'.format('control_interval', kwargs['control_interval']))
        self.timer.setInterval(self.setting.contorl_interval)
    '''

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
        """
        log.debug('write index before data generation: {} out of total {}'
                .format(self.buffer_widx, self.setting.buffer_length))
        """

        self.buffer_widx = self.buffer_widx % self.setting.buffer_length
        data = self.setting.mu + self.setting.sigma * \
                    np.random.randn(self.chunk_size, self.n_channel)
        self.buffer[self.buffer_widx:(self.buffer_widx+self.chunk_size), :] = data
        self.buffer_widx += self.chunk_size

        '''
        log.debug('write index after data generation: {} out of total'.
            format(self.buffer_widx, self.setting.buffer_length))
        '''

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
