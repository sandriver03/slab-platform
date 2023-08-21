from .Setting import Setting
from .ToolBar import ToolBar
import labplatform.stream as Stream
from labplatform.utilities.arraytools import make_dtype
from labplatform.config import get_config

from traits.api import Str, Dict, Any, Instance, Enum, Bool, Property,\
    on_trait_change, List, cached_property
from traitsui.api import Controller
from traitsui.message import error
import time
import threading

import logging
log = logging.getLogger(__name__)

Default_Stream_Params_to_Check = ('sampling_freq', 'dtype', 'shape', 'type', 'software_scaling')
Stream_Mapping = {'input': '_input_specs', 'output': '_output_specs'}


class Logic(Controller):
    """
    Master controller for both experiment and device. This is an abstract class and is not intended to be used directly
    """

    shell_variables = Dict

    # These define what variables will be available in the Python shell.  Right
    # now we can't add various stuff such as the data and interface classes
    # because they have not been created yet.  I'm not sure how we can update
    # the Python shell with new instances once the experiment has started
    # running.
    def _shell_variables_default(self):
        return dict(controller=self, c=self)

    setting         = Instance(Setting)  # parameter definition
    lock            = Instance(threading.Lock())  # thread lock or multiprocess lock
    state           = Enum('Created', 'Ready', 'Running', 'Paused', 'Stopped', 'Error')
    events          = Any  # events happened
    parent_model    = Any  # object from upper level GUI

    toolbar = Instance(ToolBar, (), toolbar=True)  # toolbar with different buttons
    name    = Property(Str, depends_on=['setting.category'])
    type    = Property(Str, depends_on=['setting.category'])
    thread  = Instance(threading.Thread)

    # if after configure the logic need to be re-initialized
    _reinitialize   = Bool(False)

    # if auto initialize when created
    _auto_initialize = Bool(True)

    # if want to use the default thread
    _use_default_thread = Bool(False)

    # if this Logic is operating in a subprocess
    _in_subprocess = Bool(False)

    # subprocess event loop and queue
    process_event_loop = Any

    # keep track of changed parameters
    _changed_params = List()

    @cached_property
    def _get_type(self):
        return self.setting.category + '_name'

    @cached_property
    def _get_name(self):
        return getattr(self.setting, self._get_type())

    def __init__(self, **metadata):
        Controller.__init__(self, model=None, **metadata)
        self.model = self

    def init(self, info):
        """
        Called when GUI is initialized; thus when used without GUI this method is not called
        """
        try:
            # store handle to model
            self.model = info.object
            self.info = info
            self.setting = self.model.setting  # link setting configuration object

            # install the toolbar
            for tb in self.trait_names(toolbar=True):
                getattr(self, tb).install(self, info)  # Device is a controller

            # should think a way to display system notifications on the GUI. Maybe a text widget?
            # or write a separate class for system notifications

            # set state
            # self.change_state('Created')

        except Exception as e:
            log.exception(e)
            self.change_state('Error')
            # display error message in GUI
            error('Error: {} when initiating. Abort.'.format(e))

    def close(self, info, is_ok):
        """
        Defines what actions to perform when closing the GUI
            :param info:
            :param is_ok:
            :return:
        """

        to_close = True
        # handle window close event when the experiment is still running
        # ask user permission if the state is not error or complete
        if self.state not in ('Error', 'Stopped', 'Created', 'Ready'):
            rtn = error('The experiment is still running, are you sure to exit?')
            if rtn:
                self.stop(info)
            else:
                to_close = False

        if to_close:
            # any cleanup needed should be put here
            self._close()

        return to_close

    def _close(self):
        """
        handles cleanup when closing the GUI
        """
        log.debug('the _close() method is not implemented')

    ''' -----------------------------------------------------------
        state related methods
    '''
    # state variables
    _initialized = Bool(False)
    _configured  = Bool(False)
    _busy        = Bool(False)   # if the Logic is busy doing something, so it cannot process function calls

    # ------------------------------------------------------
    def change_state(self, new_state=None, **kwargs):
        """
        change the state of the Logic. internal state can only be `Created`, `Ready`, `Running`, `Paused`,
        `Stopped`, or `Error`.
        other boolean state variables (_initialized, _configured, _trial_active, _pausing and _complete) can
        also be changed using syntax k=value, with k values `initialized`, `configured`, `trial_active`,
        `pausing` and `complete`.
        the method is thread safe.

        Args:
            new_state: str
            **kwargs: see description

        Returns: None

        """
        if new_state:
            with self.lock:
                self.state = new_state
            log.debug('{}: {}: state changed to {}'.format(self.setting.category,
                      self.name, self.state))

        for k in kwargs:
            with self.lock:
                setattr(self, '_' + k, kwargs[k])
            log.debug('{}: {}: state variable {} changed to {}'.
                      format(self.setting.category, self.name, '_' + k, kwargs[k]))

    # state checking methods
    def running(self):
        """Return True if the Logic is running.

        This method is thread-safe.
        """
        with self.lock:
            return self.state == 'Running'

    def configured(self):
        """Return True if the Device has already been configured.

        This method is thread-safe.
        """
        with self.lock:
            return self._configured

    def initialized(self):
        """Return True if the Device has already been initialized.

        This method is thread-safe.
        """
        with self.lock:
            return self._initialized

    def stopped(self):
        """Return True if the Device has already been stopped.

        This method is thread-safe.
        """
        with self.lock:
            return self.state == 'Stopped'

    def paused(self):
        """Return True if the Device has already been stopped.

        This method is thread-safe.
        """
        with self.lock:
            return self.state == 'Paused'

    # state change methods
    def initialize(self, **kwargs):
        """Initialize the Logic.

        This method prepares the Logic for operation by allocating memory,
        preparing devices, checking input and output specifications, etc.
        subclasses determine the behavior of this method by reimplementing
        `Logic._initialize()`.
        """
        raise NotImplementedError

    def _configure(self, **kargs):
        """
        This method is called during `Logic.configure()` and must be
        reimplemented by subclasses. Generally used to pass changes onto
        hardware it controls
        """
        return kargs

    def _initialize(self, **kargs):
        """
        This method is called during `Logic.initialize()` and must be
        reimplemented by subclasses.
        """
        raise NotImplementedError

    def _start(self):
        """
        This method is called during `Logic.start()` and must be
        reimplemented by subclasses.
        """
        raise NotImplementedError

    def _stop(self):
        """
        This method is called during `Logic.stop()` and must be
        reimplemented by subclasses.
        """
        raise NotImplementedError

    def _pause(self):
        """
        This method is called during `Logic.pause()` and must be
        reimplemented by subclasses.
        """
        raise NotImplementedError

    def _reset(self):
        """
        This method is called during `Logic.reset()` and can be
        reimplemented by subclasses.
        """
        pass

    def check_input_specs(self):
        """
        This method is called during `Device.initialize()` and may be
        reimplemented by subclasses to ensure that inputs are correctly
        configured before the node is started.

        In case of misconfiguration, this method must raise an exception.
        """
        pass

    def check_output_specs(self):
        """
        This method is called during `Device.initialize()` and may be
        reimplemented by subclasses to ensure that outputs are correctly
        configured before the node is started.

        In case of misconfiguration, this method must raise an exception.
        """
        pass

    def _after_input_connect(self, inputname):
        """
        This method is called when one of the Device's inputs has been
        connected.

        It may be reimplemented by subclasses.
        """
        pass

    def _after_output_configure(self, outputname):
        """
        This method is called when one of the Device's outputs has been
        configured.

        It may be reimplemented by subclasses.
        """
        pass

    ''' ---------------------------------------------------------------
        general methods related to preparation
    '''
    def configure(self, **kwargs):
        """
        Configure the Logic parameters.

        This method is used to set parameters defined in associated Setting
        instance. Only deals with software parameters; to interact with hardware,
        implement `_configure` method. Mainly used when running the device without
        the GUI. Otherwise, the functionality is automatically handled by the GUI.

        In Experiment instance it can also be used to configure the devices; use device name
        and a dictionary as parameters
        """
        # only allows parameters with reinit=False to be configured while the Logic is running
        if self.running() and (kwargs.keys() & self.setting.trait_names(reinit=True)) \
                and (kwargs.keys() & self.setting.trait_names(dynamic=False)):
            log.warning('Only parameters with reinit=False and dynamic=True can be configured while running')
            raise RuntimeError('Cannot configure {}: {} : the Node is running'
                               .format(self.setting.category, self.name))

        # perform parameter validation here
        paras = self._configure_validation(**kwargs)
        _subprocess_mode = False

        log.info('{}: {}: configuring...'.
                 format(self.setting.category, self.name))

        if self.setting.operating_mode == 'subprocess':
            _subprocess_mode = True

        # set parameters in Logic.setting
        for key in paras:
            if key in self.model.setting._para_list:
                setattr(self.model.setting, key, paras[key])
                log.debug('*configure- {}, {}: set attribute {} to value {}'
                    .format(self.setting.category, self.name, key, paras[key]))
            elif self.model.setting.category == 'experiment' and key in self.devices:
                self.devices[key].configure(**paras[key])
            else:
                log.warning('{}: {}: Attribute {} not exists or cannot be set. It is ignored'.
                                format(self.setting.category, self.name, key))

        # commit changes
        if self.setting.operating_mode != 'subprocess' and not _subprocess_mode:
            self.apply()
        else:
            if kwargs['operating_mode'] == 'subprocess':
                if not _subprocess_mode:
                    self.initialize()
                    self.setting_shadowcopy['operating_mode'] = 'subprocess'
                del kwargs['operating_mode']
            else:
                self.setting.operating_mode = 'subprocess'
            if self.process_event_loop and kwargs:
                self.process_event_loop.put_event(('configure', kwargs, True))
            self._pending_changes = False
            self._changed_params = []

        log.info('{}: {}: configured'.
                 format(self.setting.category, self.name))

    def _configure_validation(self, **kwargs):
        """
        validate the parameters to be configured using .configure() method
            :param kwargs: parameters defined using name=value pairs
            :return: validated parameters
        """
        return kwargs

    ''' ---------------------------------------------------------
            parameter change notification and handling

    '''
    _pending_changes = Bool(False)

    # setup the pending_changes tag so 'apply' method can be used
    @on_trait_change('model.setting.+context')
    def handle_parameter_change(self, instance, name, old, new):
        if name == 'model':
            return
        if not self.info or self.info.initialized:
            if self.model.setting.trait(name).context:
                log.debug('*handle_parameter_change- {}: {}: Attribute {} has changed from {} to {}'.
                    format(self.setting.category, self.name, name, old, new))
                # trait = instance.trait(name)
                self._changed_params.append(name)
                self.change_state(pending_changes=True)

        if self.model.setting.trait(name).reinit:
            self.change_state(reinitialize=True)

    def setting_updated(self):
        """
            This can be overriden in subclasses to implement logic for updating the
            experiment when the apply button is pressed; e.g. re-initialize memory etc.
        """
        log.debug('method setting_updated() not implemented.')

    # called when setting instance is initialized; should only happen when the controller is
    # initialized or reset
    @on_trait_change('model')
    def setting_initialized(self, instance, name, old, new):
        log.debug('model.setting changed: {}'.format(self.name))
        # link model.setting to internal attribute setting
        self.setting = self.model.setting
        self.setting_shadowcopy = self.model.setting.copy_values_to()
        self.setting_default = self.model.setting.copy_values_to()

        # for Device class, add default parameters to changed list so it can be applied
        if self.setting.category == 'device':
            self._changed_params = self.setting.get_parameters()

        # create lock
        # if not self.lock or not isinstance(self.lock, type(lock_lookup[self.setting.operating_mode]())):
            # self.lock = lock_lookup[self.setting.operating_mode]()
        if not self.lock:
            self.lock = threading.Lock()

        time.sleep(0.05)
        # set state parameters if the default configuration is ready to run
        if self._check_settings() and self._auto_initialize:
            log.debug('{}: {}: default settings are sufficient to run the device'.
                      format(self.setting.category, self.name))
            self.initialize()

    def _check_settings(self):
        ready = True
        for val in self.model.setting.get_parameters():
            if getattr(self.model.setting, val) is None:
                ready = False
                break
        return ready

    # indicate pending changes on the GUI
    @on_trait_change('_pending_changes')
    def indicate_pending_changes_on_title(self):
        if self.info and self.info.ui:
            if self._pending_changes:
                self.info.ui.title += '*'
            else:
                self.info.ui.title = self.info.ui.title.strip('*')

    ''' ----------------------------------------------------------
            GUI related methods

            these methods are linked to buttons on the GUI. can be used without GUI as well
            see ExperimentToolBar for the buttons
        '''
    setting_shadowcopy = Any  # a copy of parameters are required for using the revert button;
    # outside of the GUI this has no use
    setting_default = Any  # default values for the setting; used when resetting the device

    def apply(self, info=None):
        """
        Called when apply button is pressed or `configure` is called, committing parameter changes to hardware
        """
        # special precautions when GUI is used
        if self.info:
            if self.running() and (set(self._changed_params) & set(self.setting.trait_names(reinit=True))) \
                    and (set(self._changed_params) & set(self.setting.trait_names(dynamic=False))):
                log.warning('Only parameters with reinit=False and dynamic=True can be configured while running')
                # revert the parameters back to last commit
                self.model.setting.copy_values_from(self.setting_shadowcopy)
                self._changed_params = []
                self.change_state(pending_changes=False, reinitialize=False)
                raise ValueError('Cannot configure {}: {} : the Node is running'.
                                 format(self.setting.category, self.name))

        if self._pending_changes:
            log.debug('*apply- {}: {}: Applying requested changes'.
                      format(self.setting.category, self.name))
            try:
                self.setting_shadowcopy = self.model.setting.copy_values_to()
                self.change_state(pending_changes=False)
                # self._pending_changes = False

                # set parameters in hardware
                self._configure()

                # re-initialize when necessary
                if self._reinitialize:
                    # print('reinitializing')
                    self.initialize()

                self.setting_updated()
                # empty changed_params list
                self._changed_params = []
            except Exception as e:
                log.error(e)
                if self.info:
                    error(message='Unable to apply the changes. No changes have been made.',
                          title='Error applying changes')

        # check if parameters are sufficiently configured
        if self.state not in ('Running', 'Paused'):
            if self._check_settings():
                self.change_state(configured=True)
                if self.state != 'Ready' and self.initialized():
                    self.change_state('Ready')
            else:
                self.change_state('Created', configured=False)

    def revert(self, info=None):
        """
        Called when revert button is pressed, changes settings back to last committed state

        Note: this method only intended to work with GUI; it does not work with configure() method
        """
        log.debug('{}: {}: Reverting to last applied settings'.
                  format(self.setting.category, self.name))
        self.model.setting.copy_values_from(self.setting_shadowcopy)
        # self.pending_changes = False
        if self.setting.operating_mode != 'subprocess':
            self.apply()
        else:
            self._pending_changes = False
            self._changed_params = []
            self.process_event_loop.put_event(('revert', None))

    def reset(self, info=None):
        """
        Re-initialize the device, change everything back to default values. Any information, including data collected
        will be lost. It should not be a problem, since we don't keep data in the device object.

        It calls `stop` before rolling parameters back
        """
        if not self._in_subprocess:
            assert not self.running(), \
                'Cannot reset {}: {}: the Node is running'.format(self.setting.category, self.name)
        else:
            if self.running():
                msg = 'Cannot reset {}: {}: the Node is running'.format(self.setting.category, self.name)
                log.error(msg)
                print(msg)
                return

        log.info('attempt resetting {} {}...'.format(self.setting.category, self.name))

        if self.setting.operating_mode != 'subprocess':
            if not self.stopped():
                self.stop()
                time.sleep(0.05)

            self.model.setting.copy_values_from(self.setting_default)
            self.initialize()
            self._reset()
        # self.configure()
        else:
            self.model.setting.copy_values_from(self.setting_default)
            self.setting.operating_mode = 'subprocess'
            self._pending_changes = False
            self._changed_params = []
            self.process_event_loop.put_event(('reset', None))

        log.info('the {} {} was successfully reset'.format(self.setting.category, self.name))

    def start(self, info=None):
        """Start the Logic.

        When the Device is running it will read from its input streams and write
        to its output streams (if any). Device must be configured and initialized
        before they are started, and can be paused and restarted any number of
        times.

        Logic with pending parameter changes cannot be started.
        """
        if not self._in_subprocess:
            assert not self.stopped(), \
                'Cannot start {} {} : the Node is stopped. Reinitialize it to start again'.\
                format(self.setting.category, self.name)
            assert self.configured(), \
                'Cannot start {} {} : the Node is not configured'.\
                format(self.setting.category, self.name)
            assert self.initialized(), \
                'Cannot start {} {} : the Node is not initialized'.\
                format(self.setting.category, self.name)
            assert not self.running(), \
                'Cannot start {} {} : the Node is already running'.\
                format(self.setting.category, self.name)
            assert not self._pending_changes, \
                'Pending changes not applied on {} {}.'.\
                format(self.setting.category, self.name)
        else:
            msg = 'Cannot start {} {} :'.format(self.setting.category, self.name)
            has_error = True
            if self.stopped():
                msg = msg + 'the Node is stopped.'
            elif not self.configured():
                msg = msg + 'the Node is not configured.'
            elif not self.initialized():
                msg = msg + 'the Node is not initialized.'
            elif self.running():
                msg = msg + 'the Node is already running.'
            elif self._pending_changes:
                msg = msg + 'pending changes not applied'
            else:
                has_error = False
            if has_error:
                log.error(msg)
                print(msg)
                return

        log.info('starting {} {}...'.format(self.setting.category, self.name))

        # starting in subprocess requires different operations
        if self.setting.operating_mode == 'thread':
            # start default thread if configured
            if self._use_default_thread and not self.thread.is_alive():
                self._start_thread()
            self._start()
        elif self.setting.operating_mode == 'subprocess':
            self.process_event_loop.put_event(('start', None))
        elif self.setting.operating_mode == 'normal':
            pass
        else:
            raise ValueError('Operating mode {} not known'.format(self.setting.operating_mode))
        self.change_state('Running')

        log.info('{} {} was successfully started'.format(self.setting.category, self.name))

        if self.setting.operating_mode == 'normal':
            self._start()
            self.run_normal_mode()

    def run_normal_mode(self):
        """
        Run the logic in the normal mode. it will block the main thread, and make the Logic non-interactive with the
        console
        Returns:
            None
        """
        raise NotImplementedError

    def pause(self, info=None):
        """Pause the Logic. Paused Logic can be directly restarted
        """
        msg = 'Cannot pause {} {} : the Node is not running'.format(self.setting.category, self.name)
        if not self._in_subprocess:
            assert self.running(), msg
        else:
            if not self.running():
                log.error(msg)
                print(msg)
                return

        log.info('pausing {} {}...'.format(self.setting.category, self.name))

        if self.setting.operating_mode != 'subprocess':
            self._pause()
        else:
            self.process_event_loop.put_event(('pause', None))
        self.change_state('Paused')

        log.info('{} {} was paused'.format(self.setting.category, self.name))

    def stop(self, info=None):
        """Stop the Logic.

        This causes all input/output connections to be stopped. Logic must
        be not running before they can be stopped. Stopped Logic cannot be
        directly restarted
        """
        if not self._in_subprocess:
            assert not self.running(), \
                'Cannot stop {} {} : the Node is running'.format(self.setting.category, self.name)
        else:
            if self.running():
                msg = 'Cannot stop {} {} : the Node is running'.format(self.setting.category, self.name)
                log.error(msg)
                print(msg)
                return
        if self.stopped():
            return

        log.info('stopping {} {}...'.format(self.setting.category, self.name))

        if self.setting.operating_mode != 'subprocess':
            self._stop()
        else:
            self.process_event_loop.put_event(('stop', None))
            # do we terminate the process as well?
            self.process_event_loop.put_event(('pause_event_loop', None))
            self.process_event_loop.put_event(('stop_event_loop', None))
            self.process.join()
        '''
        for input in self.inputs.values():
            input.stop()
        for output in self.outputs.values():
            output.stop()
        '''
        self.change_state('Stopped', configured=False, initialized=False, thread_running = False)

        log.info('{} {} was stopped'.format(self.setting.category, self.name))

    '''
        threading support
    '''
    _thread_running = Bool(False)
    # can be put into _initialize so the thread is automatically created when initialize is called

    def _initialize_thread(self):
        """
        Initializes and starts the default thread associated with the Logic. Override `thread_func` to use this thread
        """
        if not self.thread or not self.thread.is_alive():
            log.debug('creating thread...')
            self.thread = threading.Thread(target=self.thread_run, daemon=False,
                                           name=self.name + '_' + 'default')
            # self.thread.start()

    # if true, add sleep(0.02) when not running to reduce CPU load; doing so will likely make response slower as well
    _thread_nonbusy_wait = Bool(True)

    def _start_thread(self):
        self.change_state(thread_running=True)
        self.thread.start()

    def _stop_thread(self):
        self.change_state(thread_running=False)
        # block until finish
        self.thread.join()

    def thread_run(self):
        """
        Function ran by the default thread. It can be paused and started indefinitely, and only runs when the state of
        the Logic is `Running`
        """
        self._prepare_thread()
        while self._thread_running:
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

            # reduce the load on waiting devices; does increase the response time though
            if self._thread_nonbusy_wait:
                time.sleep(0.02)

    def _prepare_thread(self, **kwargs):
        """
        custom thread preparation routines

        Returns:
            None
        """
        pass

    def thread_func(self):
        """
        actions to be performed in the default thread associated with the Logic
        """
        pass

    def _get_state_in_subprocess(self, state_list):
        """
        return state variables when the logic is running in a subprocess
        Args:
            state_list: list or tuple of strings; state variables to quire
        Returns:
            None; the values are put into communication queue
        """
        # does nothing when not running in subprocess
        if not self._in_subprocess or not self.process_event_loop:
            return
        res = dict()
        for var in state_list:
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
        try:
            self.process_event_loop.event_queue.put(('states', res))
        except:
            print(res)

    def get_state(self, state_list):
        """
                return state variables when the logic is running in a subprocess
                Args:
                    state_list: list or tuple of strings; state variables to quire
                Returns:
                    None; the values are put into communication queue
                """
        # does nothing when not operating in subprocess mode
        if not self.setting.operating_mode == 'subprocess':
            return
        return self.process_event_loop.get_state(state_list)

    """
        Traits View (GUI)
    """
    # Device and ExperimentLogic should have different default view

    """
    input and output using streams
    """
    input = Any  # use dict to hold all inputs
    output = Any  # use dict

    # input specifics
    _input_specs = Dict()
    # output specifics
    _output_specs = Dict()
    # the keys should match the keys in input/output dictionary
    '''
    about the _output_specs:
    it should be a dictionary of dictionaries, first level is the name of each output stream, the second level is the
    configuration of each stream. for the configuration, it should have the following keys:
        type: the type of the data, str
        shape: shape of the expected data, tuple; 1st dimension should always be 0, which is the time dimension
        sampling_freq: sampling frequency, float
        dtype: data type of the output, should be a (numpy) data type
        save: if the data coming from this stream should be saved, bool, default to True
        stream: the stream.params dictionary, created automatically when .create_output_stream is called
    '''

    def connect_input_stream(self, stream):
        """
        Args:
            stream: an Stream.OutputStream instance
        Returns:
        """
        name_str = getattr(self.setting, self.setting.category + '_name')
        if not self.input:
            self.input = dict()
        # use stream name as the key
        if stream.name in self.input.keys():
            log.warning("input stream with name: {} already exist. overriding existing one.".format(stream.name))
            # close existing input stream
            try:
                self.input[stream.name].close()
                del self.input[stream.name]
            except:
                raise
        self.input[stream.name] = Stream.InputStream()
        self.input[stream.name].connect(stream)
        # check if self params fit input params
        paras_toconfig = self._check_stream_params(stream.name, self.input)
        # if parameter mismatch
        if paras_toconfig:
            self._unmatched_input_stream_params(stream.name, self.input, paras_toconfig)
        # modify _input_specs
        # type, shape, sampling frequency, dtype, length
        stream_para = dict(type=stream.params['streamtype'], shape=(0, ) + (stream.params['shape'][1:]),
                           sampling_freq=stream.params['sampling_freq'], dtype=stream.params['dtype'])
        if self._input_specs:
            if 'dtype' in self._input_specs.keys():
                old_specs = dict()
                old_specs[name_str + '_nonstream'] = self._input_specs.copy()
        else:
            self._input_specs = dict()
        self._input_specs[stream.name] = stream_para
        # custom operations after input connects
        self._after_input_connect(stream.name)

    def _check_stream_params(self, name, stream_dict, params=Default_Stream_Params_to_Check):
        """
        check if input stream parameters match those in settings. can be overridden in subclasses

        Args:
            name: str, name of the stream to be checked
            stream_dict: dict of streams containing the stream to be checked
            params: list of str, name of the parameters to be checked

        Returns:
            dict, name and value of parameters need to be modified
        """
        paras_toconfig = dict()
        log.debug('checking parameters {} in stream {}'.format(params, name))
        paras_in_setting = self._get_stream_params(params)
        for key in paras_in_setting.keys():
            if stream_dict[name].params[key] != paras_in_setting[key]:
                paras_toconfig[key] = make_dtype(stream_dict[name].params[key])
                log.warning('stream {} parameter {} does not match corresponding parameter {} in setting'.format(
                            name, key, key))

        return paras_toconfig

    def _get_stream_params(self, params=Default_Stream_Params_to_Check):
        """
        get parameters required to correctly configure the stream from setting class
        Args:
            params: list of str, name of the parameters to get
        Returns:
            dict
        """
        paras_in_setting = {}
        for key in params:
            try:
                paras_in_setting[key] = getattr(self.setting, key)
            except AttributeError:
                # software scaling is None or 1 if not exist
                if key == 'software_scaling':
                    paras_in_setting[key] = None
                else:
                    # no such parameter in setting
                    log.warning('parameter {} not found in setting'.format(key))
        return paras_in_setting

    def _unmatched_output_stream_params(self, name, output_streams, unmatched_params):
        """
        custom, class-specific operations on unmatched output stream parameters
        Args:
            name: str, name of the stream
            output_streams: self.output, a dictionary
            unmatched_params: dict, returned by _check_stream_params
        Returns:
            None
        """
        log.warning('ummatched parameters {} in output stream {} not handled'.format(unmatched_params.keys(), name))

    def _unmatched_input_stream_params(self, name, input_streams, unmatched_params):
        """
        custom, class-specific operations on unmatched input stream parameters
        Args:
            name: str, name of the stream
            input_streams: self.input, a dictionary
            unmatched_params: dict, returned by _check_stream_params
        Returns:
            None
        """
        log.warning('ummatched parameters {} in input stream {} not handled'.format(unmatched_params.keys(), name))

    def create_output_stream(self, stream_name=None, save_data=True, **kwargs):
        """
        create a output stream, based on _output_specs and kwargs
        Args:
            stream_name: str, name of the output stream, if none use device name
            save_data: bool, if the data from this stream should be saved
            kwargs:
                see OutputStream.configure()
        Returns:
            None
        """
        name_str = getattr(self.setting, self.setting.category + '_name')
        if not stream_name:
            stream_name = name_str
        else:
            stream_name = name_str + '_' + stream_name
        stream = Stream.OutputStream(name=stream_name, node=self)
        # stream parameters, check _output_specs and kwargs
        # get dtype, type, sampling_freq and shape from setting
        params_in_setting = self._get_stream_params()
        # use params_in_setting to update kwargs, so it overwrites corresponding values in kwargs
        kwargs.update(params_in_setting)
        stream.configure(**kwargs)
        time.sleep(0.5)
        if not self.output:
            self.output = dict()
        self.output[stream_name] = stream
        # check stream parameters and settings
        paras_toconfig = self._check_stream_params(stream.name, self.output)
        # if parameter mismatch
        if paras_toconfig:
            self._unmatched_output_stream_params(stream.name, self.output, paras_toconfig)
        # modify _output_specs
        # type, shape, sampling frequency, dtype, length
        stream_para = dict(type=stream.params['type'],
                           shape=(-1,) + (stream.params['shape'][1:]),
                           software_scaling=stream.params['software_scaling'],
                           sampling_freq=stream.params['sampling_freq'],
                           dtype=stream.params['dtype'],
                           save=save_data,
                           stream=stream.params,
                           source=name_str)
        if self._output_specs:
            if 'dtype' in self._output_specs.keys():
                old_specs = dict()
                old_specs[name_str + '_nonstream'] = self._output_specs.copy()
                self._output_specs = {}
                self._output_specs.update(old_specs)
        else:
            self._output_specs = dict()
        self._output_specs[stream.name] = stream_para
        # operations after output stream creation
        self._after_output_configure(stream_name)

    def remove_stream(self, name, stream_type=None):
        """
        close and remove the named stream from the device

        Args:
            name: str, name of the stream to be closed
            stream_type: str, 'input' or 'output'. whether the stream is input or output stream. if none, search both
        Returns:
            None
        """
        if not stream_type:
            stream_type = ('input', 'output')
        if not isinstance(stream_type, (list, tuple)):
            stream_type = (stream_type, )
        for stream_name in stream_type:
            stream_dict = getattr(self, stream_name)
            if stream_dict and name in stream_dict.keys():
                log.info('removing stream: {} in group: {}'.format(name, stream_name))
                # close and delete the stream
                stream_dict[name].close()
                del stream_dict[name]
                # delete entries in the input or output specs
                specs = getattr(self, Stream_Mapping[stream_name])
                del specs[name]
