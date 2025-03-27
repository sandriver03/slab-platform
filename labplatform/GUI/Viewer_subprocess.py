from labplatform.utilities.RingBuffer import RingBuffer
from labplatform.utilities.arraytools import make_dtype
from labplatform.utilities.misc import getScreenSize
import labplatform.stream as Stream

from traits.api import HasTraits, Any, Instance, Bool, Enum
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import time
import threading
import queue
import logging
log = logging.getLogger(__name__)

default_params = dict(
    data_shape=(-1, 1),
    data_type=float,
    buffer_length=10,
    display_length=5,
    control_interval=0.1,
    data_monitor_interval=0.1,
    sampling_freq=1000,
    fig_name=None,
)


class Viewer(HasTraits):
    params = Any

    state = Enum('Created', 'Ready', 'Running', 'Paused', 'Stopped', 'Error')

    fig = Any    # window to plot figure
    data = Any   # data to be plotted
    x = Any

    data_monitor_thread = Any
    process = Any

    input = Instance(Stream.InputStream)
    _input_params = Any
    command_queue = Any

    _running = Bool(False)
    _live = Bool(False)
    _thread_running = Bool(False)

    def _default_configure(self):
        return dict(default_params)

    def configure(self, **kwargs):
        """
        Args:
            kwargs:
                data_shape: tuple
                data_type: str or a numpy data type
                buffer_length: float, length of buffer in second
                display_length: float, length of default display in second
                control_interval: float, image refresh interval in second
                data_monitor_interval: float, interval to check new data in second
                sampling_freq: float, data sampling frequency in Hz
                fig_name: str or int, name of the figure window
        Returns:
            None
        """
        if not self.params:
            self.params = self._default_configure()
        for k in kwargs:
            if k in self.params:
                self.params[k] = kwargs[k]

        shape = self.params['data_shape']
        assert shape[0] == -1 or shape[0] > 0, "First element in shape must be -1 or > 0."
        for i in range(1, len(shape)):
            assert shape[i] > 0, "Shape index %d must be > 0." % i

        self.state = 'Created'

    def configure_process(self, params):
        """
        configure the viewer instance running in the subprocess
        Args:
            params: dict, parameters to be set

        Returns:
            None
        """
        self._running = False
        if self._check_fig_exist():
            plt.close(self.params['fig_name'])

    def _init_fig(self):
        print('init figure')
        raise NotImplementedError

    def _step_fig(self):
        if not self._check_fig_exist():
            self._init_fig()
        self._update_fig()

    def _check_fig_exist(self):
        raise NotImplementedError

    def _update_fig(self):
        # print('update figure')
        raise NotImplementedError

    def _reset_fig(self):
        raise NotImplementedError

    def _process_func(self, event_q):
        self.command_queue = event_q
        self._live = True
        if self._input_params and not self.input:
            self.connect_datastream(self._input_params)
        self._setup_data_monitor_thread()
        self.data_monitor_thread.start()
        while self._live:
            self._read_process_command()
            while self._running:
                # core part of the code; read an input data point and plot it
                self._step_fig()
                self._read_process_command()
            time.sleep(0.02)

        self._process_cleanup()

    def _process_cleanup(self):
        """
        clean up after stopping the process
        Returns:
            None
        """
        # clear and join the command queue
        while True:
            try:
                self.command_queue.get(block=False)
            except queue.Empty:
                break
        self.command_queue.close()
        self.command_queue.join_thread()

    def run_in_process(self, event_q=mp.Queue()):
        # close and remove input stream
        if self.input:
            self._input_params = self.input.params
            if self.input.socket:
                self.input.close()
            self.input = None
        self.process = mp.Process(target=self._process_func, args=(event_q, ))
        self.command_queue = event_q
        self.process.start()

    def _setup_data_monitor_thread(self):
        if self.data_monitor_thread and self.data_monitor_thread.is_alive():
            log.warning('data monitor thread already running. skip the operation')
        self.data_monitor_thread = threading.Thread(target=self._data_monitor_thread_func)

    def _data_monitor_thread_func(self):
        while self._live:
            while self._thread_running:
                # print('thread running')
                # poll the input stream to receive data
                if self.input.poll():
                    _, d, _ = self.input.recv()
                    self.data.write(d)
                time.sleep(self.params['data_monitor_interval'])
            time.sleep(self.params['data_monitor_interval'])
            # self._when_paused()

    def connect_datastream(self, stream):
        """
        Args:
            stream: an Stream.OutputStream instance, or a dict containing stream parameters
        Returns:
        """
        if self.input:
            self.input.close()
            self.input = None
            time.sleep(0.1)
        self.input = Stream.InputStream()
        self.input.connect(stream)
        # check if self params fit input params
        paras_toconfig = {}
        if self.input.params['sampling_freq'] != self.params['sampling_freq']:
            paras_toconfig['sampling_freq'] = self.input.params['sampling_freq']
        if self.input.params['shape'][1:] != self.params['data_shape'][1:]:
            paras_toconfig['data_shape'] = (-1, ) + self.input.params['shape'][1:]
        if make_dtype(self.input.params['dtype']) != make_dtype(self.params['data_type']):
            paras_toconfig['data_type'] = make_dtype(self.input.params['dtype'])
        if paras_toconfig:
            self.configure(**paras_toconfig)

        self._create_data_buffer()
        # custom operations after input connects
        self._after_input_connect()
        # state
        self.state = 'Ready'

    def disconnect_stream(self):
        """
        disconnect from current stream
        Returns:
            None
        """
        if not self.input:
            return
        self.input.close()
        del self.input

    def _create_data_buffer(self):
        # create buffer to store data
        if self.params['buffer_length'] > 0:
            buffer_length_n = int(self.params['sampling_freq'] * self.params['buffer_length'])
        else:
            buffer_length_n = 1
        self.data = RingBuffer(shape=((buffer_length_n, ) + self.params['data_shape'][1:]),
                                dtype=self.params['data_type'])
        self.x = np.linspace(-self.params['buffer_length'], 0, buffer_length_n)

    def _after_input_connect(self, **kwargs):
        pass

    def _when_paused(self):
        pass

    def start(self):
        self.command_queue.put(('start', ))
        self.state = 'Running'

    def pause(self):
        self.command_queue.put(('pause', ))
        self.state = 'Paused'

    def stop(self):
        self.command_queue.put(('stop',))
        self.state = 'Stopped'

    def reset(self):
        if self.process.is_alive():
            self.command_queue.put(('reset_fig',))
        else:
            self.run_in_process()

    def get_state(self, var_list):
        self.command_queue.put(('get_state', var_list))

    def _read_command(self):
        event = self.command_queue.get()
        event_name = event[0]
        event_para = None
        para_unpack = None
        if event.__len__() > 1:
            event_para = event[1]
        if event.__len__() > 2:
            para_unpack = event[2]
        return event_name, event_para, para_unpack

    def _stop(self):
        """
        to be implemented by subclass
        Returns:
            None
        """
        pass

    def _process_command(self, command):
        if command[0] == 'start':
            print('viewer is starting')
            self._running = True
            self._thread_running = True
            self.state = 'Running'
        elif command[0] == 'pause':
            print('viewer is pausing')
            self._running = False
            self.state = 'Paused'
        elif command[0] == 'stop':
            print('viewer is stopping')
            self._running = False
            # close figure
            self._stop()
            self._thread_running = False
            self._live = False
            self.state = 'Stopped'
        elif command[0] == 'update':
            self._step_fig()
        elif command[0] == 'reset_fig':
            self._reset_fig()
        elif command[0] == 'get_state':
            self._get_state_in_subprocess(command[1])
        elif command[0] == 'disconnect':
            self.disconnect_stream()
        elif command[0] == 'connect':
            self.connect_datastream(command[1])
        elif command[0] == 'configure':
            self.configure_process(command[1])
        else:
            self._subclass_command(command)

    def _subclass_command(self, command):
        """
        sub-class may need to have some special commands, override the function when needed
        Args:
            command: tuple or list, see send_command
        Returns:
            None
        """
        log.debug('sub-class specific command not implemented')
        print('unknown command {}'.format(command[0]))

    def _read_process_command(self):
        if not self.command_queue.empty():
            command = self._read_command()
            self._process_command(command)
        # plt.pause(0.05)
        # time.sleep(0.05)

    def _get_state_in_subprocess(self, state_list):
        """
        return instance attributes when the logic is running in a subprocess
        Args:
            state_list: list or tuple of strings; state variables to quire
        Returns:
            None; the values are put into communication queue
        """
        # does nothing when not running in subprocess
        res = dict()
        for var in state_list:
            try:
                res[var] = getattr(self, var)
            except AttributeError:
                try:
                    res[var] = getattr(self, '_' + var)
                except AttributeError:
                    print('state variable {} does not exist'.format(var))
        print(res)

    def send_command(self, command):
        """
        send command to subprocess
        Args:
            command: string, tuple or list; list and string will be converted to tuple
        Returns:
            None
        """
        if isinstance(command, str):
            command = (command, )
        elif isinstance(command, list):
            command = tuple(command)
        if not isinstance(command, tuple):
            raise ValueError('data type of command is not tuple')
        self.command_queue.put(command)


class ImageViewer(Viewer):
    _new_clim = Any
    _new_cmap = Any

    def _default_configure(self):
        params = dict(default_params)
        params['fig_name'] = 'Image'
        params['data_shape'] = (-1, 1024, 1024)
        params['buffer_length'] = 0
        params['sampling_freq'] = 30
        params['clim'] = [0, 2 ** 16]
        return params

    def _init_fig(self):
        if plt.fignum_exists(self.params['fig_name']):
            # if plt.figure(self.params['fig_name']) is not self.fig:
               # self.params['fig_name'] += 1
            # else:
            plt.close(self.params['fig_name'])
        # print('create figure')
        self.fig = plt.figure(self.params['fig_name'])
        self.axes = self.fig.add_axes([0.1, 0.1, 0.85, 0.85])
        if not self.data:
            self._create_data_buffer()
        self.artists = self.axes.imshow(self.data[-1],
                                        vmin=self.params['clim'][0],
                                        vmax=self.params['clim'][1],
                                        aspect='equal',
                                        cmap='gray')
        self.fig.colorbar(self.artists, ax=self.axes)
        self.fig.show()

    def _check_fig_exist(self):
        return plt.fignum_exists(self.params['fig_name'])

    def _update_fig(self):
        if self._new_clim:
            self.artists.set_clim(self._new_clim[0], self._new_clim[1])
            self._new_clim = None
        if self._new_cmap:
            if isinstance(self._new_cmap, str):
                cmap = plt.get_cmap(self._new_cmap)
            else:
                cmap = self._new_cmap
            self.artists.set_cmap(cmap)
            self._new_cmap = None
        self.artists.set_data(self.data[-1])
        # TODO: currently it seems this steals the focus
        # https://stackoverflow.com/questions/44278369/how-to-keep-matplotlib-python-window-in-background
        # https://stackoverflow.com/questions/45729092/make-interactive-matplotlib-window-not-pop-to-front-on-each-update-windows-7
        # plt.pause(self.params['control_interval'])
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        time.sleep(self.params['control_interval'])

    def _reset_fig(self):
        pass

    def _destroy_fig(self):
        plt.close(self.params['fig_name'])

    def _stop(self):
        self._destroy_fig()

    def _after_input_connect(self, **kwargs):
        self.data.write(np.zeros(self.params['data_shape'][1:], dtype=self.params['data_type']))

    def _when_paused(self):
        plt.pause(0.1)

    def set_clim(self, clim):
        """
        set display color scale of the image
        Args:
            clim: tuple, (vmin, vmax) of the color scale to be set
        Returns:
            None
        """
        if not isinstance(clim, tuple):
            raise ValueError('parameter clim must be a tuple')
        self.send_command(('set_clim', clim))
        self.params['clim'] = clim

    def set_cmap(self, cmap):
        """
        set colormap for the image display
        Args:
            cmap: str, colormap used to display images. see matplotlib.pyplot.get_cmap; or a custom color map
        Returns:
            None
        """
        self.send_command(('set_cmap', cmap))
        self.params['clim'] = cmap

    def _subclass_command(self, command):
        if command[0] == 'set_clim':
            self._new_clim = command[1]
        if command[0] == 'set_cmap':
            self._new_cmap = command[1]
        else:
            print('command {} not known'.format(command[0]))


class DataViewer(Viewer):

    axes = Any
    artists = Any

    def _default_configure(self):
        params = dict(default_params)
        params['fig_name'] = 'Data'
        params['x_unit'] = 'time, (s)'
        params['data_unit'] = None
        params['y_lim'] = [-5, 5]
        return params

    def _init_fig(self):
        if plt.fignum_exists(self.params['fig_name']):
                plt.close(self.params['fig_name'])
        # print('create figure')
        self.fig = plt.figure(self.params['fig_name'])
        self.axes = self.fig.add_axes([0.1, 0.1, 0.85, 0.85])
        self.axes.set_xlabel(self.params['x_unit'])
        self.axes.set_ylabel(self.params['data_unit'])
        self.axes.set_ylim(-5, 5)  # since we are mostly dealing with TDT boards
        self.axes.set_xlim(-self.params['display_length'], 0.25 * self.params['display_length'])
        if not self.data:
            self._create_data_buffer()
        self.artists = self.axes.plot(self.x, self.data[:])
        self.fig.show()

    def _check_fig_exist(self):
        return plt.fignum_exists(self.params['fig_name'])

    def _update_fig(self):
        for i, p in enumerate(self.artists):
            # p.set_data(self_obj.x, self_obj.data[:, i])
            p.set_ydata(self.data[:, i])
        # plt.pause(self.params['control_interval'])
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        time.sleep(self.params['control_interval'])

    def _reset_fig(self):
        self.data.write(np.zeros(self.data.shape))

    def _after_input_connect(self, **kwargs):
        self.data.write(np.zeros(self.data.shape))

    def _when_paused(self):
        plt.pause(0.1)
