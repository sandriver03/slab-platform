"""
How to write a Device class
    you will need 3 parts for a GUI enabled Device class:
        1). DeviceToolBar class
            defines buttons on the GUI to control the action of the Device. The default one
            has Apply, Revert, Reset, Start, Pause, Resume, Terminate. If you do not need
            more buttons you can directly use the DeviceToolBar class. Otherwise subclassing
            it
        2). subclass of DeviceSetting class
            holds all configurable parameters for the Device. All parameters need to be con-
            figured should be here. See document on Experiment_setting for the structure of
            parameter definition
        3). subclass of Device class
            this is the controller class, which defines all the logic to operate the device.
            you need to at least implement _initialize, _configure, _start, _stop, _pause;
            you might want to implement setting_updated to control what happens after new
            settings are applied

"""

from labplatform.core.Setting import DeviceSetting
from labplatform.core.Logic import Logic
import labplatform.core.EventManager as EMG

from traits.api import Any, Str, Instance, Dict, Button, Either, Bool
from traitsui.api import View, Item, VGroup, Include, Tabbed, HGroup, Label

import threading
import multiprocessing as mp
import multiprocessing.connection
import logging
log = logging.getLogger(__name__)


class Device(Logic):
    """
    a prototype device, potentially using queues as input/output when running on a thread.
    """

    setting = Instance(DeviceSetting)
    type = Str

    # keep a handle to experiment
    experiment = Any
    lock = Either(threading.Lock(), mp.Lock())

    # if want to use the default thread
    # _use_default_thread = True
    _use_default_thread = True

    # subprocess event loop and queue
    process_event_loop = Instance(EMG.EventManager)
    process = Instance(mp.Process)

    # state change methods
    def initialize(self, **kwargs):
        """Initialize the Device.

        Subclass must implement `_initialize` to perform actual actions
        """

        # does not allow initializing while running
        if not self._in_subprocess:
            assert not self.running(), \
                'Cannot initialize {}: {}: the Node is running'.\
                    format(self.setting.category, self.name)
        else:
            if self.running():
                msg = 'Cannot initialize {}: {}: the Node is running'.format(self.setting.category, self.name)
                log.error(msg)
                print(msg)
                return

        log.info('{}: {}: initializing...'.
                 format(self.setting.category, self.name))

        self.check_input_specs()
        self.check_output_specs()

        if self.setting.operating_mode != 'subprocess':
            self._initialize(**kwargs)

            # create thread to monitoring hardware
            if self._use_default_thread:
                self._initialize_thread()
            if not self.lock:
                self.lock = threading.Lock()
        else:
            self._initialize_process(mp.Pipe())

        # self._reinitialize = False
        self.change_state('Created', initialized=True, reinitialize=False)

        # check if settings are good to run the device (no parameter missing)
        if not self.configured():
            if self._check_settings():
                self.change_state(pending_changes=True)
                # self.pending_changes = True
                self.configure()
        else:
            self.change_state('Ready')
            self._pending_changes = False
            self._changed_params = []

        # assign the model attribute
        if not self.model:
            self.model = self

        log.info('{}: {}: initialized'.
                 format(self.setting.category, self.name))

    def _configure(self, **kwargs):
        """
        Commits parameter changes to the controlled hardware. Must be implemented.
        Args:
            **kwargs: primary parameters defined in DeviceSetting class

        Returns:
            None
        """
        raise NotImplementedError

    """
        Traits View (GUI)
    """

    def default_traits_view(self):

        traits_view = View(
            Item('state', style='readonly', show_label=True),
            VGroup(
                Tabbed(
                    Item('setting', style='custom', show_label=False),
                    # Item('data', style='custom', show_label=False),
                    # Item('handler.tracker', style='custom'),
                ),
                Item('handler.toolbar', style='custom', show_label=False),
            ),
            resizable=True,
            title='Device: {}'.format(self.name)
        )
        return traits_view

    EditButton = Button('Config/View')

    def _EditButton_fired(self):
        if not self.info or not self.info.ui:
            self.edit_traits(handler=self)

    def get_gui_viewgroup(self):
        # the view used in ExperimentLogic GUI
        return HGroup(Label(self.name), Item('state', style='readonly'),
                      Item("EditButton", show_label=False, enabled_when="info.object.state not in ['Running']"))

    """
    support for running the device in another process
    """
    _process_running = Bool(False)

    def deinitialize(self):
        """
        release hardware and buffer, stop thread
        Returns:
            None
        """
        if not self._in_subprocess:
            assert self.state not in ('Running', 'Paused'), "de-initialization is only possible when the device is " \
                                                            "not in state of 'Running' or 'Paused'"
        else:
            if self.state in ('Running', 'Paused'):
                msg = "de-initialization is only possible when the device is not in state of 'Running' or 'Paused'"
                log.error(msg)
                print(msg)
                return
        if self.thread and self.thread.is_alive():
            self._stop_thread()
        self._deinitialize()
        self.change_state("Created")

    def _initialize_process(self, evt_q):
        """
        configure and start a copy of the device in a new process, with an event manager as event loop
        this function should be called
        """
        if self.process and self.process.is_alive():
            raise RuntimeError('A process for this device is already running')
        if self.state in ('Running', ):
            raise RuntimeError('Cannot start sub-process when the device is running. stop the device first')
        if self.state in ('Stopped', ):
            self.reset()
        self.deinitialize()
        # attach an event loop to handle device operations
        if not self.process_event_loop:
            self.process_event_loop = EMG.EventManager()
        else:
            if self.process_event_loop.state == 'Running':
                if self.process_event_loop.setting.operating_mode == 'subprocess':
                    # TODO
                    raise NotImplementedError
                else:
                    self.process_event_loop.pause()
        # make sure the event loop is not running, but configuration is good to be ran
        self.process_event_loop.configure(operating_mode='thread')
        if self.process_event_loop.event_queue:
            if hasattr(self.process_event_loop.event_queue, 'closed') \
                    and not self.process_event_loop.event_queue.closed:
                self.process_event_loop.event_queue.close()
        # register the handlers for the event loop
        self.process_event_loop._register_event_loop_controls()
        # register the handlers for the event loop to handle the device operations
        self.process_event_loop.subscribe('start', self.start)
        self.process_event_loop.subscribe('pause', self.pause)
        self.process_event_loop.subscribe('stop', self.stop)
        self.process_event_loop.subscribe('revert', self.revert)
        self.process_event_loop.subscribe('reset', self.reset)
        self.process_event_loop.subscribe('configure', self.configure)
        self.process_event_loop.subscribe('get_state', self._get_state_in_subprocess)
        # sub-class specific operations
        self.setting.operating_mode = 'thread'
        self.__initiate_process()
        # need to remove those items that cannot be serialized
        self.lock = None
        self.thread = None
        self.process = None
        self.process_event_loop.thread = None
        self.process_event_loop.lock = None
        self.process_event_loop.event_queue = None
        # make modifications if the evt_q is PipeConnection
        if isinstance(evt_q, (tuple, list)) and isinstance(evt_q[0], mp.connection.PipeConnection):
            for conn in evt_q:
                EMG.setup_PipeConn(conn)
            subprocess_Q = evt_q[1]
            parent_Q = evt_q[0]
        else:
            parent_Q, subprocess_Q = evt_q
        # setup and start a new process
        self.process = mp.Process(target=self._process_run, args=(subprocess_Q, ))
        self.process.start()
        # add un-serializable items back
        self.lock = threading.Lock()
        self.setting.operating_mode = 'subprocess'
        self.process_event_loop.lock = threading.Lock()
        self.process_event_loop.setting.operating_mode = 'subprocess'
        self.process_event_loop.event_queue = parent_Q

    def __initiate_process(self):
        pass

    def _stop_process(self):
        # make sure the device in subprocess is 'Ready', 'Paused' or 'Stopped'
        # TODO
        # send signal to the process event loop to shut it down
        self.process_event_loop.put_event(('pause_event_loop', None))
        self.process_event_loop.put_event(('stop_event_loop', None))
        self.process.join()

    def _deinitialize(self):
        """
        need to be implemented by subclasses, release hardware and buffer resource
        Returns:
            None
        """
        raise NotImplementedError

    def _process_run(self, event_q):
        """
        this method runs in the new process; need to start the event loop
        remember: everything here is NOT the same object as in the main process
        """
        self._in_subprocess = True
        self.process_event_loop._in_subprocess = True
        if isinstance(event_q, mp.connection.PipeConnection):
            EMG.setup_PipeConn(event_q)
        self.process_event_loop.event_queue = event_q
        # start the event loop
        self.process_event_loop.configure()
        self.process_event_loop.start()
