# a simulated software cam
from labplatform.core.Setting import DeviceSetting
from labplatform.core.Device import Device
from labplatform.GUI.Viewer_subprocess import ImageViewer
from labplatform.utilities.RingBuffer import RingBuffer

import numpy as np
from traits.api import Instance, Float, Any, Property, Str, CInt, Enum, Bool, List
import time
import threading
import matplotlib.pyplot as plt

import logging
log = logging.getLogger(__name__)


class shamCamSetting(DeviceSetting):

    device_type = 'ShamCam'

    control_interval = 0.02
    exposure_time = CInt(5, group='primary', dsec='exposure time, ms', camera_setting=True)
    buffer_time = Float(15, group='primary', dsec='buffered data length in second')
    shape = Any((-1, 1024, 1024), group='primary', dsec='shape of the image taken by the camera')
    shared_ram_buffer = Bool(False, group='primary', dsec='if use shared ram buffer to store camera data')
    data_mode = Enum('sequence', 'random', group='primary', dsec='debug data to be generated', reinit=False)
    data_length = CInt(-1, group='primary', dsec='if positive, the device will stop after reaching the '
                                                 'desired length. otherwise it runs indefinitely', reinit=False)
    software_scaling = CInt(2, group='primary', dsec='additional software scaling when saving data')

    dtype = Str('uint16', group='status', dsec='data type of the image. Camera property')
    type = Str('image', group='status', dsec='nature of the data')
    frame_lag = CInt(10, group='status', dsec='simulated frame delay')

    sampling_freq = Property(depends_on=['exposure_time'], group='derived',
                          dsec='frame per second of the camera, depends on exposure time')
    buffer_length = Property(depends_on=['buffer_time', 'sampling_freq'], group='derived',
                             dsec='buffer length in data points')

    def _get_sampling_freq(self):
        return 1000/(self.exposure_time + self.frame_lag)

    def _get_buffer_length(self):
        return int(self.sampling_freq * self.buffer_time)


# parameters require to reset stream
stream_params = {'shape', 'shared_ram_buffer', 'buffer_time'}


class shamCam(Device):

    cam = Any
    setting = Instance(shamCamSetting, ())
    # t0 = Float(0)
    # t1 = Float(0)
    viewer = Instance(ImageViewer)

    # do not auto-initialize
    _auto_initialize = False

    # buffer to hold snapshot
    buffer_snapshot = Any
    # buffer to hold live stream
    buffer = Instance(RingBuffer)
    # camera buffer
    buffer_cam = Any
    # for now, to detect if a new frame is arrived, check if the buffer has changed;
    # in the future should change to use callbacks
    _frameCnt = Any
    _cam_frameCnt = Any   # simulate frame number at the camera side
    _frameTms = List  # time stamp for each frame

    _stream_para_changed = Bool(False)

    _data_info = Any
    _data_thread = Instance(threading.Thread)
    _cam_running = Bool(False)
    # if camera live mode is on
    _live_on = Bool(False)
    # if in preview mode
    _preview = Bool(False)

    def _initialize(self, **kwargs):
        # set cam handle to self
        self.cam = self
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
        # simulate cam buffer
        self.buffer_snapshot = np.zeros(self.setting.shape[1:], dtype=self.setting.dtype)
        self.buffer_cam = np.zeros(self.setting.shape[1:], dtype=self.setting.dtype)
        # actual software buffer
        if self._stream_para_changed:
            if self.buffer:
                del self.buffer
        if not self.buffer:
            self.buffer = RingBuffer(shape=((self.setting.buffer_length, ) + self.setting.shape[1:]),
                                     dtype=self.setting.dtype,
                                     double=False)
        # parameters used to check new frames
        self._frameCnt = 0
        self._cam_frameCnt = 0
        self._frameTms = []
        self._data_info = (0, time.time())
        # setup image viewer
        if self._stream_para_changed and self.viewer:
            self.viewer.stop()
            self.viewer = None
        if not self.viewer:
            self.viewer = ImageViewer()
            self.viewer.configure(fig_name=self.name + '_Image',
                                  data_monitor_interval=1/self.setting.sampling_freq,
                                  data_shape=self.setting.shape)
            self.viewer._input_params = self.output['ShamCam_0_tcp'].params.copy()
            self.viewer.run_in_process()
        self._stream_para_changed = False

    def _reset_frameInfo(self):
        self._frameCnt = 0
        self._frameTms = []

    def _start(self):
        self.start_live_cb()

    def _stop(self):
        self.change_state(cam_running=False, live_on=False)
        # self.viewer.stop()

    def _pause(self):
        # if trigger mode is not internal, then pause should be handled by trigger signal
        self.stop_live()

    def _configure(self):
        pass

    def _reset(self):
        self.viewer.reset()

    def snapshot(self):
        """
        take a single image with the camera. only works when 'Ready' or 'Paused'
        Returns:

        """
        if self.state not in ('Ready', 'Paused'):
            if self.state == 'Running':
                raise Exception('the Camera {} is running. Cannot take snapshot while running.')
            elif self.state == 'Created':
                raise Exception('the Camera {} is not initialized. Please initialize it first'
                                .format(self.getName()))
            elif self.state == 'Stopped':
                raise Exception('the Camera {} is already stopped'.format(self.getName()))
            else:
                raise Exception('Cannot take snapshot in current state: {}'.format(self.state))

        if self._cam_running and self._live_on:
            self.buffer_snapshot = self.buffer_cam.copy()
        else:
            self.buffer_snapshot = self.cam.get_frame(self.setting.exposure_time).copy()

    def get_frame(self, exposure_time=None):
        if not self._live_on:
            # generate a single frame
            self._single_frame()
            return self.buffer_cam
        else:
            raise RuntimeError('cannot take a single frame when the camera is in live mode')

    def start_live_cb(self):
        if not self._cam_running:
            self.change_state(cam_running=True)
        if not self._data_thread or not self._data_thread.is_alive():
            self._data_thread = threading.Thread(target=self._simulate_frame)
            self._data_thread.start()
        if not self._live_on:
            self.change_state(live_on=True)

    def get_live_frame_cb(self):
        return self.buffer_cam

    def stop_live(self):
        self.change_state(live_on=False)

    def get_frameCnt(self):
        return self._data_info

    def _simulate_frame(self):
        while self._cam_running:
            while self._live_on:
                if time.time() - self._data_info[1] > (self.setting.exposure_time + self.setting.frame_lag)/1000:
                    # generate new frame
                    self._single_frame()
                time.sleep(0.01)
            time.sleep(0.1)

    def _single_frame(self):
        """
        generate single frame
        Returns:
            None
        """
        if self.setting.data_mode == 'sequence':
            self.buffer_cam = self._cam_frameCnt * np.ones(self.setting.shape[1:],
                                                           dtype=self.setting.dtype)
        elif self.setting.data_mode == 'random':
            if np.issubdtype(self.setting.dtype, np.integer):
                self.buffer_cam = np.random.randint(2 ** 16, size=self.setting.shape[1:],
                                                    dtype=self.setting.dtype)
            else:
                self.buffer_cam = np.random.randn(self.setting.shape[1], self.setting.shape[2])
        else:
            raise ValueError('data mode {} is not known'.format(self.setting.data_mode))
        # record frame number and frame time
        self._cam_frameCnt += 1
        self._data_info = (self._cam_frameCnt, time.time())

    def show_snapshot(self, img_name=None):
        """
        display saved snapshot frame
        Returns:
            None
        """
        if not img_name:
            img_name = self.name + '_snapshot'
        fig = plt.figure(img_name)
        fig.imshow(self.buffer_snapshot)

    def preview(self):
        """
        live view from the camera, without saving the frames
        should only work in 'Ready' or 'Paused' state

        Returns:
            None
        """
        if self.state not in ('Ready', 'Paused'):
            raise Exception("the preview only works when the camera is in 'Ready' or 'Paused' state")

        if not self._live_on:
            # setup camera to live mode
            self.cam.start_live_cb()
            time.sleep(0.05)
            if self.viewer.state in ('Ready', 'Paused'):
                self.viewer.start()
            elif self.viewer.state == 'Running':
                pass
            else:
                raise RuntimeError('viewer with state {} cannot be started'.format(self.viewer.state))

        self._preview = True
        self.start()

    def stop_preview(self):
        self._preview = False
        self.cam.stop_live()
        self.pause()

    def toggle_liveview(self):
        if self.viewer.state is 'Running':
            self.viewer.pause()
        elif self.viewer.state in ['Ready', 'Paused']:
            self.viewer.start()
        else:
            log.warning("live view not available in current viewer state: {}"
                        .format(self.viewer.state))

    def thread_func(self):
        # decide if new frame has arrived
        # new_frame_val = self.buffer_cam[self.__frame_check_idx[0], self.__frame_check_idx[1]]
        frame_Cnt = self.cam.get_frameCnt()
        if frame_Cnt[0] != self._frameCnt:
            # new frame arrived
            new_frame = self.cam.get_live_frame_cb()
            if not self._preview and self.setting.buffer_time > 0:
                self.buffer.write(new_frame)
            self.output[self.name + '_tcp'].send(new_frame, header=(b'd',))
            self._frameCnt = frame_Cnt[0]
            self._frameTms.append((frame_Cnt[0], frame_Cnt[1]))
            if frame_Cnt[0] - self._frameCnt > 1:
                log.warning("{} frame(s) skipped".format(frame_Cnt[0] - self._frameCnt))

        if self.setting.data_length > 0:
            self.stop_at_datalength()

    def stop_at_datalength(self):
        if self.buffer._written_index >= self.setting.data_length:
            data = np.zeros(self.setting.shape[1:],
                            self.setting.dtype)
            self.output['ShamCam_0_tcp'].send(data, header=(b's',))
            self.pause()
            if self.experiment:
                self.experiment().process_event({'trial_stop': 0})


"""
code for profiling
"""
if __name__ == '__main__':
    import time
    import cProfile
    scam = shamCam()
    scam.configure(exposure_time=15)
    time.sleep(1)
    scam.viewer.start()


    def _single_run(cam, frame_count, sleep_time=0.01):
        frame_Cnt = cam.get_frameCnt()
        if frame_Cnt[0] != cam._frameCnt:
            new_frame = cam.get_live_frame_cb()
            if not cam._preview:
                cam.buffer.write(new_frame)
            cam.output[cam.name + '_tcp'].send(new_frame)
            cam._frameCnt = frame_Cnt[0]
            cam._frameTms.append((frame_Cnt[0], frame_Cnt[1]))
            if frame_Cnt[0] - cam._frameCnt > 1:
                log.warning("{} frame(s) skipped".format(frame_Cnt[0] - cam._frameCnt))

            frame_count += 1
        time.sleep(sleep_time)
        return frame_count

    def sim_start(cam, frame_to_capture=1000, sleep_time=0.01):
        cam.start_live_cb()
        cam._preview = False
        frame_count = 0
        while frame_count < frame_to_capture:
            frame_count = _single_run(cam, frame_count, sleep_time)

        cam.stop_live()

    # cProfile.run('sim_start(scam)')

    """
    # test speed of writing data to h5 file
    import tables as tl
    t_all = []
    for i in range(10):
        fh = tl.open_file('test.h5', mode='w', )
        t0 = time.time()
        fh.create_earray(fh.root, name='scam_data', obj=scam.buffer.buffer[:1, :, :])
        t1 = time.time()
        fh.root.scam_data.append(scam.buffer.buffer)
        t2 = time.time()
        t_all.append((t2 - t0, t2 - t1))
        fh.close()
    """
