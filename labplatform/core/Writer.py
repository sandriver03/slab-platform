from labplatform.utilities import H5helpers as h5t
# from labplatform.utilities.arraytools import make_dtype
from labplatform.config import get_config
import labplatform.stream as Stream

import numpy as np
import tables as tl
from collections import OrderedDict
from collections.abc import Iterable
import threading
import queue
import multiprocessing as mp
import time


import logging
log = logging.getLogger(__name__)

# TODO: force_stop not working


def update_experiment_history(where, exp_name, record_name='experiment_history',
                              close_on_finish=True, **kwargs):
    """
    Update experiment history record (a table.Table node) in the HDF file. Either add a new entry to the table if
    record about current experiment is not there, or update existing records. Close the file after finish.

    Args:
        where: Tables.Group, the node whose attributes need to be set
              can also be a tuple or list of str, in form of (file_path, node_path) which can be used for function
              H5helpers.get_node_handle
        exp_name: str or byte array, current experiment name
        record_name: str, name of the experiment record as a attribute
        close_on_finish: if close the files after finish. default True
        **kwargs: See Core.Subject.subject_history_obj for individual field

    Returns:
        None
    """
    if isinstance(where, (tuple, list)):
        where = h5t.get_node_handle(where[0], where[1])
    if isinstance(exp_name, str):
        exp_name = exp_name.encode()

    history_record = getattr(where, record_name)
    idx = h5t.find_table_index(history_record, "experiment_name == exp_name",
                               {'exp_name': exp_name})
    if not idx:
        log.info('Record for experiment: {} not found in file {}. Creating new entry...'.
                 format(exp_name, where))
    h5t.update_table(history_record, idx, close_on_finish, **kwargs)


def set_h5node_attrs(where, close_on_finish=True, **kwargs):
    """
    Add node attribute set to both the master data file as well as the external file

    Args:
        where: Tables.Group, the node whose attributes need to be set
              can also be a tuple or list of str, in form of (file_path, node_path) which can be used for function
              H5helpers.get_node_handle
        close_on_finish: if close the master file after finish. default True
        **kwargs: values to be added or modified to the node attribute set

    Returns:
        None
    """
    if isinstance(where, (tuple, list)):
        where = h5t.get_node_handle(where[0], where[1])

    for k, v in kwargs.items():
        try:
            where._v_attrs[k] = v
        except tl.exceptions.HDF5ExtError:
            log.debug('attribute {} is too large for tables._v_attrs. save it as tables.array'.format(k))
            where._v_file.create_array(where, name=k, obj=v)
    if close_on_finish:
        where._v_file.close()


def prepare_trial_storage(data_file_path, data_spec, current_trial, trial_duration=0):
    """
    prepare external file node to store data in each trial
    Args:
        data_file_path: tuple of strings, absolute path and node path for data file
        data_spec: dict or OrderedDict, data specifications
        current_trial: int, current trial number
        trial_duration: float, in s
    Returns:
        None
    """
    ext_data_file = h5t.get_file_handle(data_file_path[0])
    # create node in the external data file
    node_name = 'trial_' + str(current_trial).zfill(get_config('FILL_DIGIT'))
    trial_node = ext_data_file.create_group(data_file_path[1], node_name, node_name)

    trial_store = OrderedDict()
    trial_store['trial_number'] = current_trial

    # create individual parameters as either a FileChannel or MemoryChannel instance
    if data_spec:
        for v in data_spec:
            # if this data need to be saved
            if not v['save']:
                continue

            trial_store[v['name']] = v['store'](node=trial_node, fs=v['metadata']['fs'],
                                                dtype=v['metadata']['dtype'], name=v['name'],
                                                data_shape=v['metadata']['shape'],
                                                source=v['metadata']['source'])
            '''
            set data length: 1) if possible, get it from device._output_specs
                             2) else, calculate from trial_duration and sampling frequency
                             3) else, use default (in device.setting, there is expected_duration)
            '''
            if v['type'] == 'Ram':
                if 'data_length' in v['metadata']:
                    trial_store[v['name']]._data_length = v['metadata']['data_length']
                elif trial_duration > 0:
                    trial_store[v['name']]._data_length = \
                        int(trial_duration * v['metadata']['fs'])
                else:
                    trial_store[v['name']]._data_length = 0
            else:
                pass
            # create buffer
            trial_store[v['name']].create_buffer()
    else:
        log.warning('Data storage is empty, no data from the devices is saved')
        # raise ValueError('Data storage to be prepared not specified! Please run method get_data_template()')

    # add event log for the trial
    fh = trial_node._v_file
    description = np.dtype([('ts', np.float64), ('event', 'S512')])
    node = fh.create_table(trial_node, 'event_log', description)
    trial_store['event_log'] = node
    # add trial log for the trial
    description = np.dtype([('key', 'S512'), ('value', 'S512')])
    node = fh.create_table(trial_node, 'trial_log', description)
    trial_store['trial_log'] = node
    # a flag indicating if there is data not saved
    trial_store['pending_changes'] = False
    # a handle to the data file
    trial_store['file_handle'] = trial_node._v_file
    return trial_store


class WriterMixin:
    """
    common methods needed for different writer classes. cannot be used alone
    assuming mixed in class has the following attributes:
        inputs (dict),
        data_specs (dict or OrderedDict)
    """

    def __init__(self, **kwargs):
        if 'inputs' in kwargs and kwargs['inputs']:
            self.inputs = kwargs['inputs']
        else:
            self.inputs = dict()
        if 'data_specs' in kwargs:
            self.data_specs = kwargs['data_specs']
        else:
            self.data_specs = dict()
        if 'master_file' in kwargs:
            self._master_file = kwargs['master_file']
        else:
            self._master_file = None   # tuple of strings, absolute path and node path for data file
        if 'subject_file' in kwargs:
            self._subject_file = kwargs['subject_file']
        else:
            self._subject_file = None  # tuple of strings, absolute path and node path for subject file
        self._mf_handle = None     # temporally hold the master file handle when needed
        self._sf_handle = None     # temporally hold the subject file handle
        if 'external_file' in kwargs:
            self._data_file = kwargs['external_file']
        else:
            self._data_file = None     # tuple of strings, absolute path and node path for external data file
        self._df_handle = None     # temporally hold the external data file handle when needed
        if 'experiment_name' in kwargs:
            self.experiment_name = kwargs['experiment_name']
        else:
            self.experiment_name = None
        if 'input_names' in kwargs:
            self.input_names = kwargs['input_names']
        else:
            self.input_names = None
        # flag array to indicate which input has finished for current trial
        if self.input_names:
            self._input_finished_flags = np.ones(self.input_names.__len__())
        else:
            self._input_finished_flags = None

        self.current = OrderedDict()
        self.all_trials = []
        self.last_trial = OrderedDict()    # hold information for last trial
        self.buffer = None   # in case a buffer at worker level is needed
        self.name = 'Writer worker: ' + self.experiment_name   # name of the worker instance
        # status flags
        self._trial_finished = True  # if saving current trial data is finished
        self._current_trial = 0       # index of current trial
        self._current_file_handle = None   # h5 file handle
        # connect all input streams
        self._stream_connected = False    # if data streams are connected
        if kwargs['writer_type'] in ('plain', 'thread') and kwargs['connect_streams']:
            self.connect_streams()
            self._stream_connected = True

    def prepare_trial_storage(self, current_trial, trial_duration=0):
        """
        Prepare data storage for each trial. `self.data_spec` must already been set.
        """
        self.last_trial = self.current
        log.info("prepare trial storage for trial: {}".format(current_trial))
        self.current = prepare_trial_storage((self._data_file[0], self._data_file[1]),
                                             self.data_specs,
                                             current_trial,
                                             trial_duration)
        self._current_file_handle = self.current['file_handle']
        self._current_trial = current_trial
        self._input_finished_flags = np.zeros(self.input_names.__len__())
        self._trial_finished = False

    def connect_streams(self, reconnect=False):
        """
        connect to all streams in .data_specs
        Args:
            reconnect: bool, if reset the connection to the input when it already exists

        Returns:
            None
        """
        for item in self.data_specs:
            if item['stream']:
                self.connect_datastream(item['stream'], item['name'], reconnect)

    def connect_datastream(self, stream, name, reconnect=False):
        """
        Args:
            stream: an Stream.OutputStream instance, or a dict containing stream parameters
            name: str, name of the stream
            reconnect: bool, if reset the connection to the input when it already exists
        Returns:
            None
        """
        if self.inputs and name in self.inputs:
            if reconnect:
                self.inputs[name].close()
                del self.inputs[name]
                time.sleep(0.1)
            else:
                return
        self.inputs[name] = Stream.InputStream(name=name)
        self.inputs[name].connect(stream)
        self._after_input_connect()

    def _after_input_connect(self):
        pass

    def input_finished(self, current_trial, input_names):
        """
        set input(s) to be finished in _input_finished_flags
        Args:
            current_trial: int, trial index
            input_names: list or tuple of str, name of inputs to be set
        Returns:
            None
        """
        if current_trial == self._current_trial:
            for nm in input_names:
                self._input_finished_flags[self.input_names.index(nm)] = 1
        else:
            self._cached_events.append(('set_input_finish', (current_trial, input_names)))

    def _all_input_finished(self):
        """
        check if all inputs have finished for current trial
        Returns:
            Bool
        """
        return np.all(self._input_finished_flags)

    def disconnect_stream(self, name):
        """
        disconnect from the stream given by name
        Args:
            name: str, name of the stream
        Returns:
            None
        """
        if self.inputs and name in self.inputs:
            self.inputs[name].close()
            del self.inputs[name]

    def write(self, key, data, current_trial):
        """
        Write data into specific data storage specified by key

        Args:
            key: str, name of the data storage to write to
            data: data to be written; should be a numpy.Array
            current_trial: int, trial index

        Returns:
            None
        """
        if current_trial == self._current_trial:
            self.current[key].write(data)
        else:
            self._cached_events.append(('write', (key, data, current_trial)))

    def log_event(self, ts, event, current_trial):
        """
        Add event entry in to event log.

        Args:
            ts: trial time when event occurred; float
            event: event name, string
            current_trial: int, current trial index

        Returns:
            None
        """
        # The append() method of a tables.Table class requires a list of rows
        # (i.e. records) to append to the table.  Since we only append a single
        # row at a time, we need to nest it as a list that contains a single
        # record.
        if current_trial == self._current_trial:
            self.current['event_log'].append([(ts, event)])
        else:
            self._cached_events.append(('log_event', (ts, event, current_trial)))

    def log_trial(self, current_trial, info_dict):
        """
        Add kwargs items into trial log as base strings

        Args:
            current_trial: int, current trial index
            info_dict: dictionary of information to be logged; taking from kwargs in the caller

        Returns:
            None
        """

        # This is a very inefficient implementation (appends require
        # reallocating information in memory).
        if current_trial == self._current_trial:
            for k, v in info_dict.items():
                info = [(k, v)]
                self.current['trial_log'].append(info)
        else:
            self._cached_events.append(('log_trial', (current_trial, info_dict)))

    def update_experiment_history(self, info_dict):
        """
        Update experiment history record (a table.Table node) in the HDF file. Either add a new entry to the table if
        record about current experiment is not there, or update existing records. Close the file after finish by
        default. if do not want to close the file on finish, provide argument close_on_finish in kwargs

        Args:
            info_dict: dictionary of information to be logged; taking from kwargs in the caller

        Returns:
            None
        """
        exp_name = self.experiment_name.encode()
        # master file
        update_experiment_history(self._master_file,
                                  exp_name,
                                  **info_dict)
        # subject file
        update_experiment_history(self._subject_file,
                                  exp_name,
                                  **info_dict)

    def set_experiment_attrs(self, info_dict):
        """
        Add node attribute set to both the master data file as well as the external file. by default. closes the master
        file but keeps the external file open

        Args:
            info_dict: dictionary of information to be logged; taking from kwargs in the caller

        Returns:
            None
        """
        # master file
        m_node = h5t.get_node_handle(self._master_file[0], self._master_file[1]+'/'+self._master_file[2])
        set_h5node_attrs(m_node, close_on_finish=True, **info_dict)
        # external file
        if self._data_file:
            node = h5t.get_node_handle(self._data_file[0], self._data_file[1])
            set_h5node_attrs(node, close_on_finish=False, **info_dict)

    def save(self, save_range='current', close_file=True):
        """
        Called by stop_experiment when the stop button is pressed.  This is your chance to save relevant data.

        Note: only MemoryStorage need to be explicitly saved
        """
        if save_range == 'all' and self.all_trials:
            data = self.all_trials
        else:
            data = self.current

        # write data into H5 file
        if data:
            if isinstance(data, OrderedDict):
                self.save_current(data, close_file)
            else:  # all trials; data is a list
                for v in data:
                    if v['pending_changes']:
                        self.save_current(v, close_file)

    @staticmethod
    def save_current(data, close_file):
        """
        Save current trial

        Args:
            data: self.current, an OrderedDict
            close_file: bool, if close the file after saving

        Returns:
            None
        """
        for dk, dv in data.items():
            if hasattr(dv, 'save'):
                dv.save()
        if 'pending_changes' in data:
            data['pending_changes'] = False
        # flush the file if possible
        data['file_handle'].flush()
        if close_file:
            data['file_handle'].close()


class AsyncMixin:
    """
    common methods needed for different async (threaded or subprocessed) writer classes. cannot be used alone
    assuming mixed in class has the following attributes:
        command_q (queue.Queue or multiprocessing.Queue),
    """

    def __init__(self, command_q, response_q=None, **kwargs):
        self.command_q = command_q
        self.response_q = response_q

        # status flags
        self._running = True
        self._alive = True
        self._is_stopped = False
        self._trial_finished = True  # if saving current trial data is finished
        self._current_trial = 0  # index of current trial
        self._ready = False   # if the subprocess loop is ready

        # event handling setting
        self._cached_events = []  # events staged for later processing
        if 'max_evt_per_iteration' in kwargs:
            self._max_evt_per_iteration = kwargs['max_evt_per_iteration']
        else:
            self._max_evt_per_iteration = 4
        self.handlers = dict()
        self.resp_handlers = dict()
        self.stream_handlers = dict()

        if 'control_interval' in kwargs:
            self.control_interval = kwargs['control_interval']
        else:
            self.control_interval = 0.1
        self.state = 'Running'

        # register handlers received through kwargs
        # handlers should be received as name: func pairs in a dict
        if 'stream_handler' in kwargs:
            for stream_name, handler in kwargs['stream_handler'].items():
                self.register_stream_handler(stream_name, handler)
        if 'handler' in kwargs:
            for evt_name, handler in kwargs['handler'].items():
                self.register_handler(evt_name, handler)
        if 'resp_handler' in kwargs:
            for resp_name, handler in kwargs['resp_handler'].items():
                self.register_response_handler(resp_name, handler)

        # register default event handlers
        self.register_handler('disconnect', self.disconnect_stream)
        self.register_handler('connect', self.connect_datastream)
        self.register_handler('prepare_next_trial', self.async_prepare_trial_storage)
        self.register_handler('log_event', self.log_event)
        self.register_handler('log_trial', self.log_trial)
        self.register_handler('update_history', self.update_experiment_history)
        self.register_handler('set_experiment_attrs', self.set_experiment_attrs)
        self.register_handler('write', self.write)
        self.register_handler('save', self.save)
        self.register_handler('set_input_finish', self.input_finished)
        self.register_handler('print', self._print)
        self.register_handler('stop', self.stop)
        self.register_handler('force_stop', self.force_stop)
        self.register_handler('get_response', self.generate_response)
        self.register_handler('reg_handler', self.register_handler)
        self.register_handler('reg_resp_handler', self.register_response_handler)
        self.register_handler('reg_stream_handler', self.register_stream_handler)
        # register handlers to get response
        self.register_response_handler('get_state', self.get_state)

    def get_state(self, key):
        return getattr(self, key)

    # todo: how to decide if one trial is finished?
    def async_prepare_trial_storage(self, current_trial, trial_duration):
        """
        in async environment, we must check if current trial has already finished to start next trial
        """
        self._trial_finished = self._all_input_finished()
        if self._trial_finished:
            self.prepare_trial_storage(current_trial, trial_duration)
        else:
            log.debug("current trial: {} has not finished. caching event...".format(self._current_trial))
            self._cached_events.append(('prepare_next_trial', (current_trial, trial_duration)))

    def register_handler(self, evt_name, handler):
        if evt_name not in self.handlers.keys():
            self.handlers[evt_name] = []
        self.handlers[evt_name].append(handler)

    def register_response_handler(self, resp_name, handler):
        if resp_name not in self.resp_handlers.keys():
            self.resp_handlers[resp_name] = []
        self.resp_handlers[resp_name].append(handler)

    def register_stream_handler(self, stream_name, handler):
        """
        stream handler should always use self as first input argument. it will be automatically added when calling the
        handlers
        each stream should only have one handler
        Args:
            stream_name: str, name of the stream to be handled by the handler
            handler: python callable
        Returns:
            None
        """
        self.stream_handlers[stream_name] = handler

    def process_command(self, command):
        """
        Args:
            command: a tuple with format (command_name(str), params(any))
        Returns:
            None
        """
        if command[0] == 'start':
            log.info('starting the data writer')
            self._running = True
            self.state = 'Running'
        elif command[0] == 'pause':
            log.info('pausing the data writer')
            self._running = False
            self.state = 'Paused'
        else:
            self._process_evt(command)

    def stop(self):
        if self._cached_events or not self._all_input_finished():
            log.debug('data writing not finished. stop later...')
            self._cached_events.append(('stop', 0))
        else:
            log.info('stopping the data writer')
            self._running = False
            self._stop()
            self.subroutine_clean_up()
            self._alive = False
            self.state = 'Stopped'

    def _stop(self):
        pass

    def force_stop(self):
        """
        force a stop on the writer, ignore unfinished events and inputs
        Returns:
            None
        """
        log.info('force stopping the data writer...')
        self._running = False
        self._stop()
        self.subroutine_clean_up()
        self._alive = False
        self.state = 'Stopped'

    def generate_response(self, command):
        # debugging
        # print(command)
        if command[0] in self.resp_handlers:
            # get all handlers for the command
            for h in self.resp_handlers[command[0]]:
                resp = self._process_command(command[1], h)
                try:
                    self.response_q.put((command, resp))
                except TypeError as e:        # AuthenticationString cannot be pickled
                    print('The return value(s) cannot be pickled. The value(s) is: {}'.format(resp))

    @staticmethod
    def _process_command(command, handler):
        try:
            if not command:
                resp = handler()
            elif isinstance(command, Iterable):
                if isinstance(command, (tuple, list, str)):
                    resp = handler(*command)
                elif isinstance(command, dict):
                    resp = handler(**command)
                else:
                    resp = None
                    msg = "handling parameter class: {} not implemented".format(command.__class__)
                    log.warning(msg)
                    print(msg)
            else:
                resp = handler(command)
            return resp
        except Exception as e:
            print(e)

    def _process_evt(self, evt):
        # events for which no response is needed
        if evt[0] in self.handlers:
            for h in self.handlers[evt[0]]:
                try:
                    resp = self._process_command(evt[1], h)
                except IndexError as e:
                    print(e)
        # events that some response is required (through response_queue)
        elif evt[0] in self.resp_handlers:
            for h in self.resp_handlers[evt[0]]:
                try:
                    resp = self._process_command(evt[1], h)
                    if resp is not None:
                        try:
                            self.response_q.put((evt, resp))
                        except TypeError as e:  # AuthenticationString cannot be pickled
                            print('The return value(s) cannot be pickled. The value(s) is: {}'.format(resp))
                    else:
                        print("command {} with parameter {} returned None".format(evt[0], evt[1]))
                        self.response_q.put((evt, None))
                except IndexError as e:
                    print(e)
        else:
            print('event with name: {} has no handler associated with'.format(evt[0]))

    def _print(self, **kwargs):
        """
        used to test the event processing
        """
        for k, v in kwargs.items():
            print('{}: {}'.format(k, v))

    def subroutine_clean_up(self):
        """
        clean up subroutine (thread or subprocess) when terminating
        need to: close all the sockets
                 close and join all the queues
        """
        # close all sockets
        if self.inputs:
            for nm, stream in self.inputs.items():
                stream.close()
        # save data if it is not saved (file is still open)
        try:
            self.save()
        except tl.ClosedFileError:
            pass
        # close file if necessary
        if self._current_file_handle.isopen:
            self._current_file_handle.close()
        # TODO: do we need to close queues here?

    def run(self):
        """
        the thread and subprocess can use the same run method
        """
        # initial setup
        # connect input streams; _stream_connected comes from WriterMixin
        if not self._stream_connected:
            self.connect_streams()
            self._stream_connected = True   # _stream_connected comes from WriterMixin
        self._ready = True
        # loop
        while self._alive:
            if not self.command_q.empty():
                self.process_command(self.command_q.get())

            while self._running:
                t0 = time.time()

                # first go over cached events to see if any of these can be processed
                cached_event_count = len(self._cached_events)
                for idx in range(cached_event_count):
                    self.process_command(self._cached_events.pop(0))

                # process command coming from command_q
                evt_count = 0
                while not self.command_q.empty() and evt_count < self._max_evt_per_iteration:
                    try:
                        command = self.command_q.get()
                        self.process_command(command)
                        evt_count += 1
                    except AttributeError as e:         # occurs because some obj cannot be passed to subprocess
                        print(e)

                # read data from streams and write it to file
                if self.inputs:
                    for name, stream in self.inputs.items():
                        # each stream should only have one handler
                        try:
                            self.stream_handlers[name](self, stream)
                        except KeyError:
                            print('key error trying to get stream handler')
                            msg = 'stream: {} has no handler to handler its data'.format(name)
                            log.warning(msg)
                            print(msg)

                # idle time
                t = time.time() - t0
                if t < self.control_interval:
                    time.sleep(self.control_interval - t)
                else:
                    log.warning('{}: Loop execution time longer than control interval ({} vs. {}). '.
                                format(self.name, t, self.control_interval))

            time.sleep(self.control_interval)

        self._is_stopped = True

    def read_stream_data(self, stream, N_packet=None):
        """
        Need to already have setup input streams connecting output streams in .data_spec.
        Ideally need to read multiple packets; number of packets to read is decided by stream's sampling frequency and
        reader's control interval

        Args:
            stream: Stream.InputStream instance, the stream to read from
            N_packet: number of packet to read; if None read all available if the stream is monitorred, otherwise read 1

        Returns:
            data read
        """
        pass


class StreamWriterThread(WriterMixin, AsyncMixin, threading.Thread):
    """
    receive data from zmp sockets and write it into h5 file, with pytables backend
    this class runs from a thread
    """

    def __init__(self, command_q, response_q=None, **kwargs):
        """
        Args:
            command_q: a queue.Queue instance to send command to the thread
            response_q: a queue.Queue instance to receive response from the thread
            kwargs:
                see WriterMixin and AsyncMixin
        """
        threading.Thread.__init__(self, name='ThreadedDataWriter')
        WriterMixin.__init__(self, **kwargs)
        AsyncMixin.__init__(self, command_q, response_q=response_q, **kwargs)


class PlainWriter(WriterMixin):

    def __init__(self, command_q, response_q=None, **kwargs):
        WriterMixin.__init__(self, **kwargs)
        self._cached_events = []


class StreamWriterProcess(WriterMixin, AsyncMixin, mp.Process):
    """
    a server which receives data from zmp sockets and write it into h5 file, with pytables backend
    this class runs from a separate process
    """

    def __init__(self, command_q, response_q=None, **kwargs):
        """
        Args:
            command_q: a multiprocessing.Queue class to send command to the process
            response_q: a multiprocessing.Queue instance to receive response from the process
            kwargs:
                see WriterMixin and AsyncMixin
        """
        mp.Process.__init__(self, name='ProcessedDataWriter')
        WriterMixin.__init__(self, **kwargs)
        AsyncMixin.__init__(self, command_q, response_q=response_q, **kwargs)


class WriterWrapper:

    def __init__(self, command_q, response_q=None):
        self.worker = None
        self.command_q = command_q
        self.response_q = response_q
        self._worker_monitor_thread = None
        self.name = None

    # start and clean up worker
    def stop_writer(self, block=False):
        # need some clean up work
        self.command_q.put(('stop', 0))
        # wait for the worker to stop
        self.worker.join(timeout=0.1)
        # check if the writer has stopped
        if self.worker.is_alive():
            # worker is still working on the data writing, need to wait
            if block:
                self._worker_monitor()
            else:
                self._worker_monitor_thread = threading.Thread(target=self._worker_monitor)
                self._worker_monitor_thread.start()
        else:
            # close command queue
            self._clear_queue()

    def force_stop(self):
        # force stopping
        self.command_q.put(('force_stop', 0))
        # wait for the worker to stop
        self.worker.join(timeout=2)
        if self.worker.is_alive():
            # worker is still running, there is some error
            log.error('Worker: {} cannot be stopped!'.format(self.name))
        else:
            self._clear_queue()

    def _worker_monitor(self):
        while self.worker.is_alive():
            log.debug('data writer is still writing data, wait...')
            print('waiting for data writer to finish data writing')
            time.sleep(1)
        print('writer is stopped')
        self._clear_queue()

    def _clear_queue(self):
        raise NotImplementedError

    def input_finished(self, current_trial, finished_inputs):
        """
        set input(s) to be finished in _input_finished_flags
        Args:
            current_trial: int, trial index
            finished_inputs: list or tuple of str, name of inputs to be set
        Returns:
            None
        """
        self.command_q.put(('set_input_finish', (current_trial, finished_inputs, )))

    def prepare_trial_storage(self, current_trial, trial_duration=0):
        self.command_q.put(['prepare_next_trial', [current_trial, trial_duration]])

    def connect_datastream(self, stream, name, reconnect=False):
        self.command_q.put(['connect', (stream, name, reconnect)])

    def disconnect_stream(self, name):
        self.command_q.put(('disconnect', (name, )))

    def write(self, key, data, current_trial):
        """
        Write data into specific data storage specified by key

        Args:
            key: str, name of the data storage to write to
            data: data to be written; should be a numpy.Array
            current_trial: int, trial number for the data
        Returns:
            None
        """
        self.command_q.put(('write', (key, data, current_trial)))

    def log_event(self, ts, event, current_trial):
        """
        Add event entry in to event log.

        Args:
            ts: trial time when event occurred; float
            event: event name, string
            current_trial: int, trial index

        Returns:
            None
        """
        # The append() method of a tables.Table class requires a list of rows
        # (i.e. records) to append to the table.  Since we only append a single
        # row at a time, we need to nest it as a list that contains a single
        # record.
        self.command_q.put(('log_event', (ts, event, current_trial)))

    def log_trial(self, current_trial, **kwargs):
        """
        Add kwargs items into trial log as base strings

        Args:
            current_trial: int, trial index
            **kwargs: anything to be logged; max length 30

        Returns:
            None
        """
        self.command_q.put(('log_trial', (current_trial, kwargs)))

    def update_experiment_history(self, **kwargs):
        """
        Update experiment history record (a table.Table node) in the HDF file. Either add a new entry to the table if
        record about current experiment is not there, or update existing records. Close the file after finish by
        default. if do not want to close the file on finish, provide argument close_on_finish in kwargs

        Args:
            **kwargs: See Core.Subject.subject_history_obj for individual field

        Returns:
            None
        """
        self.command_q.put(('update_history', (kwargs, )))

    def set_experiment_attrs(self, **kwargs):
        """
        Add node attribute set to both the master data file as well as the external file. by default. closes the master
        file but keeps the external file open

        Args:
            **kwargs: values to be added or modified to the node attribute set

        Returns:
            None
        """
        self.command_q.put(('set_experiment_attrs', (kwargs, )))

    def save(self, save_range='current', close_file=True):
        """
        Called by stop_experiment when the stop button is pressed.  This is your chance to save relevant data.

        Note: only MemoryStorage need to be explicitly saved
        """
        self.command_q.put(('save', (save_range, close_file)))

    def get_response(self, command, time_out=0.2):
        """
        get response from the worker subroutine
        Args:
            command: tuple or list in form of (str, *params)
            time_out: time out for the response to arrive
        Returns:
            the response from the subtoutine
        """
        self.command_q.put(('get_response', (command, )))
        # wait for response with time out
        try:
            return self.response_q.get(timeout=time_out)
        except queue.Empty:
            log.warning('trying to get response with command: {} timed out'.format(command))


class SyncPlainWriter:
    """
    plain writer which does not perform async operations
    """

    def __init__(self, command_q=None, response_q=None, **kwargs):
        self.worker = PlainWriter(command_q=command_q, response_q=response_q, **kwargs)
        if 'experiment_name' in kwargs:
            self.name = "Sync writer: " + kwargs['experiment_name']
        else:
            self.name = "Sync writer"
        self._params = kwargs

    # start and clean up worker
    def stop_writer(self):
        # need some clean up work
        print('writer is stopped')
        # self.save(save_range='all', close_file=True)

    def _clear_queue(self):
        raise NotImplementedError

    def input_finished(self, current_trial, finished_inputs):
        """
        set input(s) to be finished in _input_finished_flags
        Args:
            current_trial: int, trial index
            finished_inputs: list or tuple of str, name of inputs to be set
        Returns:
            None
        """
        pass

    def prepare_trial_storage(self, current_trial, trial_duration=0):
        self.worker.prepare_trial_storage(current_trial, trial_duration)

    def connect_datastream(self, stream, name, reconnect=False):
        self.worker.connect_datastream(stream, name, reconnect)

    def disconnect_stream(self, name):
        self.worker.disconnect_stream(name)

    def write(self, key, data, current_trial):
        """
        Write data into specific data storage specified by key

        Args:
            key: str, name of the data storage to write to
            data: data to be written; should be a numpy.Array
            current_trial: int, trial number for the data
        Returns:
            None
        """
        self.worker.write(key, data, current_trial)

    def log_event(self, ts, event, current_trial):
        """
        Add event entry in to event log.

        Args:
            ts: trial time when event occurred; float
            event: event name, string
            current_trial: int, current trial index

        Returns:
            None
        """
        # The append() method of a tables.Table class requires a list of rows
        # (i.e. records) to append to the table.  Since we only append a single
        # row at a time, we need to nest it as a list that contains a single
        # record.
        self.worker.log_event(ts, event, current_trial)

    def log_trial(self, current_trial, **kwargs):
        """
        Add kwargs items into trial log as base strings

        Args:
            current_trial: int, current trial index
            **kwargs: anything to be logged; max length 30

        Returns:
            None
        """
        self.worker.log_trial(current_trial, kwargs)

    def update_experiment_history(self, **kwargs):
        """
        Update experiment history record (a table.Table node) in the HDF file. Either add a new entry to the table if
        record about current experiment is not there, or update existing records. Close the file after finish by
        default. if do not want to close the file on finish, provide argument close_on_finish in kwargs

        Args:
            **kwargs: See Core.Subject.subject_history_obj for individual field

        Returns:
            None
        """
        self.worker.update_experiment_history(kwargs)

    def set_experiment_attrs(self, **kwargs):
        """
        Add node attribute set to both the master data file as well as the external file. by default. closes the master
        file but keeps the external file open

        Args:
            **kwargs: values to be added or modified to the node attribute set

        Returns:
            None
        """
        self.worker.set_experiment_attrs(kwargs)

    def save(self, save_range='current', close_file=True):
        """
        Called by stop_experiment when the stop button is pressed.  This is your chance to save relevant data.

        Note: only MemoryStorage need to be explicitly saved
        """
        self.worker.save(save_range, close_file)


class ThreadedStreamWriter(WriterWrapper):

    def __init__(self, command_q, response_q=None, **kwargs):
        WriterWrapper.__init__(self, command_q, response_q=response_q)
        if 'experiment_name' in kwargs:
            self.name = "Threaded writer: " + kwargs['experiment_name']
        else:
            self.name = "Threaded writer"
        self.worker = StreamWriterThread(command_q, response_q=response_q, **kwargs)
        self._params = kwargs
        self.worker.start()

    def _clear_queue(self):
        while True:
            try:
                self.command_q.get(block=False)
            except queue.Empty:
                break
        self.command_q.unfinished_tasks = 0
        self.command_q.join()


class ProcessStreamWriter(WriterWrapper):

    def __init__(self, command_q, response_q=None, **kwargs):
        WriterWrapper.__init__(self, command_q, response_q=response_q)
        if 'experiment_name' in kwargs:
            self.name = "Process writer: " + kwargs['experiment_name']
        else:
            self.name = "Process writer"
        self.worker = StreamWriterProcess(command_q, response_q=response_q, **kwargs)
        self._params = kwargs
        self.worker.start()

    def _clear_queue(self):
        while True:
            try:
                self.command_q.get(block=False)
            except queue.Empty:
                break
        self.command_q.close()
        self.command_q.join_thread()


writer_types = {'plain': SyncPlainWriter, 'thread': ThreadedStreamWriter, 'subprocess': ProcessStreamWriter}
