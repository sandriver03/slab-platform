"""
a naive implementation with traits.Event. could run into race condition or lock condition. need to be tested if used in
a complex situation, e.g. close-loop behavior experiment
order of event handling is not clear/undefined. use queue and thread for better control of event orders.
"""

from labplatform.core.Logic import Logic
from labplatform.core.Setting import DeviceSetting

from traits.api import HasTraits, Event, Dict, Float, Instance, Either, Bool, Any, List
from queue import PriorityQueue, Queue

import logging
import threading
import multiprocessing as mp
import multiprocessing.connection
import queue
import time
log = logging.getLogger(__name__)


class TraitEventManager(HasTraits):

    managed_client = Dict()

    def subscribe(self, event_name, client, handler):
        """
        only subscribe to a 'event' type trait. if the event does not exist, create it
        :param event_name: str
        :param client: a HasTrait object in which the event resides
        :param handler: a function or method which is called when the event fires
        :return: none
        """
        if not isinstance(client, HasTraits) or not isinstance(event_name, str):
            raise AttributeError('provided arguments are not expected type')
        # create the event trait if not already exist
        if not client.trait(event_name):
            client.add_trait(event_name, Event)
        else:
            if not client.trait(event_name).type == 'event':
                log.warning('request attribute: {} is not an event. only subscribe to event is allowed'.
                            format(event_name))
                raise ValueError('request attribute {} is not a trait event'.format(event_name))
        if client not in self.managed_client:
            self.managed_client[client] = []
        if (event_name, handler) not in self.managed_client[client]:
            client.on_trait_change(handler, event_name)
            self.managed_client[client].append((event_name, handler))
        else:
            log.warning('event-handler pair {}-{} already exists. No changed applied'.
                        format(event_name, handler))

    def unsubscribe(self, event_name, client, handler):
        if (event_name, handler) not in self.managed_client[client]:
            log.warning('event-handler pair {}-{} does not exist. No changed applied'.
                        format(event_name, handler))
        else:
            # get dynamically added handlers associated with the particular event
            hs = client.trait(event_name)._notifiers(0)
            for idx, h in enumerate(hs):
                if h.equals(handler):
                    del hs[idx]
            self.managed_client[client].remove((event_name, handler))

    def get_handlers(self, client=None):
        if client is None:
            client = [key for key in self.managed_client.keys()]
        if isinstance(client, (list, tuple)):
            res = {}
            for c in client:
                if c not in self.managed_client.keys():
                    log.warning('requested client: {} is not managed by this manager'.format(client))
                else:
                    res[c] = self.managed_client[c]
            return res
        else:
            if client not in self.managed_client.keys():
                log.warning('requested client: {} is not managed by this manager'.format(client))
                return None
            else:
                return self.managed_client[client]


"""
event manager using queue and thread
"""


def put_event(event, event_queue, priority=3):
    """

    Args:
        event: a tuple with three elements, first is a str indicating event name, second is argument(s) to the event
            handler, third is if the argument(s) need to be unpacked.
        priority: priority of the event, smaller number indicates higher priority
        event_queue: the queue in which the event is put

    Returns:
        None
    """
    # make sure event is a tuple of 2 elements
    assert isinstance(event, tuple), "event must be a tuple"
    event_queue.put((priority, event))


class EventManagerSetting(DeviceSetting):

    device_type = 'EventManager'
    control_interval = 0.05


_default_event_params = {'state': 'All', 'when_busy': 'ignore'}
_possible_param_vals = {'state': ['Created', 'Ready', 'Running', 'Paused', 'Stopped', 'Error'],
                        'when_busy': ['discard', 'cache']}


class EventManager(Logic):

    """
    the manager uses a PriorityQueue to transfer the events. each element in the queue will be a tuple of
    (Priority, event), in which the event should contain what the handler needs
    event should be a tuple of (event_name, kw_args)
    the event manager can also be used as trial timer and/or experiment timer
    important: always use hardware controls for precise timing/event processing.
    """
    setting = Instance(EventManagerSetting, ())
    managed_events = Dict()
    managed_handlers = Dict()
    event_queue = Either(PriorityQueue(), Queue(), mp.connection.PipeConnection)
    start_time_local = Float
    start_time_global = Float
    local_time = Float(0)   # so it can also be used as a timer
    global_time = Float(0)   # 2 separate clocks allow global (e.g. experiment) vs local (e.g. trial) control
    _discard_event_when_pause = Bool(True)  # if throw away events arrived when paused
    _current_event = Any  # currently being processed event

    _use_default_thread = True

    # cached events
    _cached_events = List()
    # managed device; when used to control a device in subprocess
    _managed_device = Any

    process = Instance(mp.Process)
    # debug variable
    child_event_q = Any

    # TODO: use weakref for handler?
    def subscribe(self, event_name, handler, **kwargs):
        """
        bind event_name to handler(s). handlers should be python callables. for the same event, handlers will be called
        by the order they are added
        :param event_name: str
        :param handler: a function or method which is called when the event fires
        :kwargs:
            state: state limits on when the event should be processed; str of allowed states in the Logic, or 'All'
            when_busy: what to do when currently event processing is not possible; 'discard' to throw away the
                       event, or 'cache' to keep the event until processing is possible
        :return: none
        """
        params = _default_event_params.copy()
        for kw in kwargs:
            if kw in params:
                if kwargs[kw] in _possible_param_vals[kw]:
                    params.update({kw: kwargs[kw]})
                else:
                    raise ValueError('value {} for parameter {} is not allowed'.format(kwargs[kw], kw))
        if not hasattr(handler, '__call__') or not isinstance(event_name, str):
            raise AttributeError('provided arguments are not expected type')
        if event_name not in self.managed_events:
            self.managed_events[event_name] = [[], {}]
        if handler not in self.managed_handlers:
            self.managed_handlers[handler] = [[], {}]
        if handler not in self.managed_events[event_name][0]:
            self.managed_events[event_name][0].append(handler)
            self.managed_events[event_name][1] = params
            self.managed_handlers[handler][0].append((event_name, params))
            self.managed_handlers[handler][1] = params
        else:
            log.warning('event-handler pair {}-{} already exists. Update event parameters'.
                        format(event_name, handler))
            if self.managed_events[event_name][1] != params:
                self.managed_events[event_name][1] = params
                self.managed_handlers[handler][1] = params
            # event_list = self.managed_events[event_name][0]
            # self.managed_events[event_name][event_list.index(handler)] = handler
            # name_list = [hp_pairs[0] for hp_pairs in self.managed_handlers[handler]]
            # self.managed_handlers[handler][name_list.index(event_name)] = (event_name, params)

    def unsubscribe(self, event_name, handler):
        if handler not in self.managed_events[event_name][0]:
            log.warning('event-handler pair {}-{} does not exist. No changed applied'.
                        format(event_name, handler))
        else:
            # get dynamically added handlers associated with the particular event
            self.managed_events[event_name][0].remove(handler)
            self.managed_handlers[handler][0].remove(event_name)

    def get_handlers(self, event_names=None):
        if event_names is None:
            event_names = [key for key in self.managed_events.keys()]
        if isinstance(event_names, (list, tuple)):
            res = {}
            for c in event_names:
                if c not in self.managed_events.keys():
                    log.info('requested event: {} is not managed by this manager'.format(c))
                else:
                    res[c] = self.managed_events[c]
            return res
        else:
            if event_names not in self.managed_events.keys():
                log.info('requested event: {} is not managed by this manager'.format(event_names))
                return None
            else:
                return self.managed_events[event_names]

    # state change methods
    def initialize(self, **kwargs):
        """Initialize the Device.

        Subclass must implement `_initialize` to perform actual actions
        """

        '''
        # stop the Device if it is running
        if self.running():
            self.pause()
            self.stop()
        '''
        if not self.lock:
            self.lock = threading.Lock()

        # does not allow initializing while running
        if not self._in_subprocess:
            assert not self.running(), \
                'Cannot initialize {}: {}: the Node is running'.\
                    format(self.setting.category, self.name)
        else:
            if not self.running():
                msg = 'Cannot initialize {}: {}: the Node is running'.format(self.setting.category, self.name)
                log.error(msg)
                print(msg)
                return

        log.info('{}: {}: initializing...'.
                 format(self.setting.category, self.name))

        self.check_input_specs()
        self.check_output_specs()
        self._initialize(**kwargs)

        # create thread to monitoring hardware
        if self._use_default_thread:
            self._initialize_thread()

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

        # assign the model attribute
        if not self.model:
            self.model = self

        log.info('{}: {}: initialized'.
                 format(self.setting.category, self.name))

    def _initialize(self, **kargs):
        if not self.event_queue and self.setting.operating_mode is not 'subprocess':
            self.event_queue = PriorityQueue()
        else:
            log.info('inter-process communication method is not provided')

    def _configure(self, **kwargs):
        pass

    def _start(self):
        pass

    def _pause(self):
        pass

    def _stop(self):
        # clear the event queue
        self._clear_eventQ()

    def _deinitialize(self):
        pass

    def _clear_eventQ(self):
        """
        clear all items from the event queue
        Returns:
        """
        while not self.event_queue.empty():
            event = self.event_queue.get()[1]
            if isinstance(self.event_queue, (queue.Queue, queue.PriorityQueue)):
                self.event_queue.task_done()
            log.debug('discard event with name {} due to clear event queue'.format(event[0]))

    # process events from event_queue
    def thread_func(self):
        # print('event loop running')
        if not self.event_queue.empty():
            self.change_state(busy=True)
            event = self._read_event()
            self._handle_event(event)
            self.change_state(busy=False)
        elif self._cached_events:
            self._handle_event(self._cached_events.pop(0))
        else:
            pass

    def _read_event(self):
        self.change_state(busy=True)
        event = self.event_queue.get()[1]
        event_name = event[0]
        event_para = None
        para_unpack = None
        if event.__len__() > 1:
            event_para = event[1]
        if event.__len__() > 2:
            para_unpack = event[2]
        log.debug('get event with name {}'.format(event_name))
        return event_name, event_para, para_unpack

    def _handle_event(self, event):
        event_name, event_para, para_unpack = event
        if event_name in self.managed_events:
            # check state to see if the event need to be handled
            self_state = self.state if not self._managed_device else self._managed_device.state
            valid_cond = self.managed_events[event_name][1]
            if valid_cond['state'] == self_state or valid_cond['state'] == 'All':
                log.debug('handling event')
                for h in self.managed_events[event_name][0]:
                    try:
                        if event_para:
                            if para_unpack:
                                if isinstance(event_para, dict):
                                    h(**event_para)
                                else:
                                    h(*event_para)
                            else:
                                h(event_para)
                        else:
                            h()
                    except Exception as e:
                        print('\033[91m', e)
                    log.debug("processed event {} with handler {}".format(event_name, h))
            elif valid_cond['when_busy'] == 'cache':
                self._cached_events.append(event)
        else:
            log.warning("event with name: {} is not known. the event is lost".format(event_name))
        # queue sync, indicating current task finishes
        if isinstance(self.event_queue, (queue.Queue, queue.PriorityQueue)):
            self.event_queue.task_done()

    def thread_run(self):
        """
        need two operation mode: if events arrived when 'Paused' is held or discarded
        """
        while self._thread_running:
            # update timer time
            self.local_time = time.time() - self.start_time_local
            self.global_time = time.time() - self.start_time_global
            while self.state == 'Running':
                t_track = time.time()
                self.thread_func()
                exec_time = time.time() - t_track
                if exec_time > self.setting.control_interval:
                    log.warning('{}: Loop execution time longer than control interval ({} vs. {}). '
                                'Try increase control interval'.format(self.name, exec_time,
                                                                       self.setting.control_interval))
                else:
                    time.sleep(self.setting.control_interval - exec_time)

            # we are here when the event manager is Paused
            if not self.event_queue.empty():
                event = self._read_event()
                self._handle_event(event)

            # when in subprocess
            if self._in_subprocess and not self.event_queue.empty():
                pass

            # reduce the load on waiting devices; does increase the response time though
            if self._thread_nonbusy_wait:
                time.sleep(0.02)

    def _reset_local_time(self):
        with self.lock:
            self.start_time_local = 0

    def _reset_global_time(self):
        with self.lock:
            self.start_time_global = 0

    def _reset_all_time(self):
        self._reset_local_time()
        self._reset_global_time()

    def start(self, info=None):
        self.start_time_local = 0
        self.start_time_global = 0
        Logic.start(self, info)

    """
    timer setup
    """
    def setup_local_timer(self, timeout, handler):
        pass

    def setup_global_timer(self, timeout, handler):
        pass

    """
    multi-process support
    """
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

    # run the event manager in a different process
    def _initialize_process(self, evt_q):
        if self.process and self.process.is_alive():
            raise RuntimeError('A process for this device is already running')
        if self.state in ('Running', ):
            raise RuntimeError('Cannot start sub-process when the device is running. stop the device first')
        if self.state in ('Stopped', ):
            self.reset()
        if self.state in ('Ready', 'Paused'):
            # release hardware resources
            self.deinitialize()
        # setup and start a new process
        self.event_queue = None
        self.lock = None
        self.thread = None
        # make modifications if the evt_q is PipeConnection
        if isinstance(evt_q, (tuple, list)) and isinstance(evt_q[0], mp.connection.PipeConnection):
            for conn in evt_q:
                setup_PipeConn(conn)
            subprocess_Q = evt_q[1]
            parent_Q = evt_q[0]
        else:
            parent_Q, subprocess_Q = evt_q
        # register a couple of handlers to operate itself in the sub-process
        self._register_event_loop_controls()
        self.subscribe('get_state', self._get_state_in_subprocess)
        self.process = mp.Process(target=self.process_func, args=(subprocess_Q,))
        self.process.start()
        self.lock = threading.Lock()
        self.event_queue = parent_Q
        self.change_state('Running')
        # Debug
        # self.child_event_q = subprocess_Q
        self.process_event_loop = self

    def _register_event_loop_controls(self):
        self.subscribe('stop_event_loop', self.stop)
        self.subscribe('pause_event_loop', self.pause, state='Running')
        self.subscribe('start_event_loop', self.start)
        self.subscribe('get_eventloop_state', self._get_eventloop_state_in_subprocess)

    def _stop_event_loop(self):
        pass

    def process_func(self, q):
        # code to be run in the sub-process
        self.lock = threading.Lock()
        self.configure()
        self.process_event_loop = self
        if isinstance(q, mp.connection.PipeConnection):
            setup_PipeConn(q)
        self._in_subprocess = True
        self.event_queue = q
        self.start()

    def put_event(self, event, priority=3):
        put_event(event, self.event_queue, priority)

    def get_state(self, var_list):
        self.put_event(('get_state', var_list, False))
        if not self.event_queue.empty(0.1):
            return self.event_queue.get()

    def get_eventloop_state(self, var_list):
        self.put_event(('get_eventloop_state', var_list, False))
        if not self.event_queue.empty(0.1):
            return self.event_queue.get()

    def _get_eventloop_state_in_subprocess(self, var_list):
        if not self._in_subprocess:
            return
        res = dict()
        for var in var_list:
            if var == 'settings':
                res[var] = self.setting.get_parameter_value()
            else:
                try:
                    res[var] = getattr(self, var)
                except AttributeError:
                    try:
                        res[var] = getattr(self, '_' + var)
                    except AttributeError:
                        log.warning('state variable {} does not exist'.format(var))
        self.event_queue.put(('states', res))


def setup_PipeConn(PipeConn):
    """
    add methods 'empty', 'put',  'get' to the object

    Args:
        PipeConn: a PipeConnection object returned by multiprocessing.Pipe()

    """

    def empty(self, timeout=0.0):
        return not self.poll(timeout)

    def put(self, obj):
        self.send(obj)

    def get(self):
        return self.recv()

    PipeConn.empty = empty.__get__(PipeConn)
    PipeConn.put = put.__get__(PipeConn)
    PipeConn.get = get.__get__(PipeConn)

