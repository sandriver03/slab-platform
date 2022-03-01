from labplatform.config import get_config
from labplatform.core.Device import Device
from labplatform.core.Setting import DeviceSetting
from labplatform.core import TDTblackbox as tdt

from traits.api import CFloat, CInt, Str, Any, Instance
import os

import logging

log = logging.getLogger(__name__)


class RP2_Ole_Setting(DeviceSetting):
    sampling_freq = CFloat(48288.125, group='primary', dsec='sampling frequency of the device (Hz)')
    buffer_size_max = CInt(50000, group='status', dsec='buffer size cannot be larger than this')
    file = Str('RCX\\BitLight.rcx', group='primary', dsec='name of the rcx file to load')
    processor = Str('RP2', group='status', dsec='name of the processor')
    connection = Str('USB', group='status', dsec='')
    index = CInt(1, group='primary', dsec='index of the device to connect to')
    stimulus = Any(group='primary', dsec='stimulus to play', reinit=False)


class RP2_Ole_Device(Device):
    setting = RP2_Ole_Setting()
    handle = Any

    def _initialize(self, **kwargs):
        expdir = get_config('DEVICE_ROOT')
        self.handle = tdt.initialize_processor(processor=self.setting.processor, connection=self.setting.connection,
                                               index=self.setting.index, path=os.path.join(expdir, self.setting.file))

        self._output_specs = {}

    def _configure(self, **kargs):
        if self.setting.stimulus.__len__():
            # self.handle.WriteTagV('datain', 0, self.setting.stimulus)
            self.handle.SetTagVal('playbuflen', len(self.setting.stimulus))

    def _start(self):
        self.handle.SoftTrg(1)

    def _pause(self):
        pass

    def _stop(self):
        self.handle.Halt()

    def thread_func(self):
        while self.handle.GetTagVal('playback'):
            pass

        if self.experiment:
            self.pause()
            self.experiment.process_event({'trial_stop': 0})


if __name__ == "__main__":
    import numpy as np

    RP2 = RP2_Ole_Device()
    RP2.setting.stimulus = np.ones(int(RP2.setting.sampling_freq * 1))
    RP2.initialize()
    RP2.configure()
    RP2.start()
