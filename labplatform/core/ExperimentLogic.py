from .Setting import ExperimentSetting
from .Logic import Logic
from .ToolBar import ToolBar
from .Subject import Subject
from labplatform.config import get_config

from traits.api import Str, Dict, Any, Instance, Bool, on_trait_change
from traitsui.message import error
from traitsui.api import Item, VGroup, View, Tabbed
import time
from threading import Timer  # threading.Timer cannot be repeatedly called; however should be sufficient here
import datetime
import os
import weakref
from collections import OrderedDict
from copy import deepcopy

import logging

log = logging.getLogger(__name__)

DATETIME_FMT = get_config('DATETIME_FMT')
DATE_FMT = get_config('DATE_FMT')
TIME_FMT = get_config('TIME_FMT')


class ExperimentLogic(Logic):
    """
    Controller class for running experiments
    """

    trial_timer = Instance(Timer)
    timer       = Instance(Timer)
    internal_timer = Instance(Timer)
    data        = Any  # all the information need to be saved should be here
    subject     = Instance(Subject)  # subject class
    devices     = Dict()  # instances of Device (sub)class
    performance = Any  # 1st order performance to check quality of experiment

    toolbar = Instance(ToolBar, (), toolbar=True)  # toolbar with different buttons
    setting = Instance(ExperimentSetting)

    # these should be passed down from the gateway function
    computer             = Str(dsec='The computer the experiment is running on')
    experimenter         = Str(dsec='The person who runs the experiment')

    _verbose     = Bool(False)   # level of logging
    # this dict will be saved as node attribute in h5 data file when experiment starts
    _tosave_para = Dict()

    # init need to be rewritten to take care of subject
    def init(self, info):
        """
        Called when GUI is initialized
        """
        super(ExperimentLogic, self).init(info)

        # any other way of getting subject information?
        if not self.subject:
            try:
                self.subject = self.parent_model.subject
            except AttributeError as e:
                log.error(e)
                self.change_state('Error')
                # display error message in GUI
                error('Error: {} when trying to get experiment subject. Abort.'.format(e))


    # make sure the subject cannot be changed once created
    @on_trait_change('subject')
    def handle_subject_change(self, instance, name, old, new):
        if old:
            log.error('Attempt changing subject during experiment')
            raise ValueError('Subject cannot be changed once the experiment starts!')

    # state change methods
    def initialize(self, **kwargs):
        """Initialize the ExperimentLogic.

        This method prepares the Logic for operation by allocating memory, preparing devices, checking input and output
        specifications, etc. Subclasses determine the behavior of this method by reimplementing `_initialize`.

        Important: Once the experiment is ran, calling this method again will stop current experiment (by calling `
        stop`) and start a new experiment
        """
        # stop the logic if it is running
        assert not self.running(), \
            'Cannot initialize {}: {}: it is running'.\
            format(self.setting.category, self.name)

        log.info('{}: {}: initializing...'.
                format(self.setting.category, self.name))

        # initialize the devices
        for kv, dv in self.devices.items():
            if not dv.initialized():
                dv.initialize()
            dv.experiment = weakref.ref(self)

        # initialize data instance
        if self.data:
            # if experiment already ran, save old data by calling stop() and start a new experiment
            if self._started:
                if self.state != 'Stopped':
                    self.stop()
                self.data = self.data.__class__()
                self.data.initialize(self.setting.experiment_name, self.subject)
            # otherwise, continue with current experiment
            elif not self.data.store_node_name:  # data has not been initialized
                self.data.initialize(self.setting.experiment_name, self.subject)
            else:  # already initialized
                pass
        else:
            raise ValueError('Data instance is not present')

        self._initialize(**kwargs)

        # create thread to monitoring hardware
        if self._use_default_thread:
            self._initialize_thread()

        # reset current trial to 0
        self.setting.current_trial = 0
        # reset state variables
        self.change_state('Created', initialized=True, complete=False, pausing=False,
                          trial_active=False, reinitialize=False)
        # self._reinitialize = False

        # check if settings are good to run the experiment (no parameter missing)
        if not self.configured():
            if self._check_settings():
                self.change_state(pending_changes=True)
                self.configure()
        else:
            self.change_state('Ready')

        # assign the model attribute
        if not self.model:
            self.model = self

        log.info('{}: {}: initialized'.
                format(self.setting.category, self.name))

    def reset(self, info=None):
        """
        Resetting experiment after it is ran will setup a new experiment. current experiment will be saved using stop()
        method
        """
        assert not self.running(), \
            'Cannot reset {}: {}: it is running'.\
            format(self.setting.category, self.name)

        if self.state not in ('Created', 'Ready'):
            log.warning('Resetting a ran experiment will stop current experiment and start a new one')

        # reset all devices
        for kv, dv in self.devices.items():
            dv.reset()

        # reset experiment parameters
        Logic.reset(self, info)

        # reset state variables specific for ExperimentLogic
        self.change_state(complete=False, ran=False, pausing=False, trial_active=False)

    def start(self, info=None):
        """
        Handles starting an experiment (called when the start button is pressed).

        Can only start experiment when state is either 'Ready' or 'Paused'. Subclasses must implement
        `setup_experiment`, `start_experiment`, and possibly implement `configure_experiment`

        Order of method execution when `start` is called:
            `before_start` -> `**setup_experiment**` -> `*configure_experiment*` -> `store_info_when_start` ->
            `**start_experiment**`
        """
        assert self.configured(), \
            'Cannot start {} {} : the Node is not configured'. \
                format(self.setting.category, self.name)
        assert self.initialized(), \
            'Cannot start {} {} : the Node is not initialized'. \
                format(self.setting.category, self.name)
        if self.state not in ['Ready', 'Paused']:
            # Don't attempt to start experiment, it is already running or has
            # stopped.
            return

        self._before_start_validate()

        try:
            if self.state == 'Ready':
                log.info('starting {} {}...'.
                    format(self.setting.category, self.name))

                # setup_experiment should load the necessary circuits and initialize
                # the buffers. This data is required before the hardware process is
                # launched since the shared memory, locks and pipelines must be
                # created.
                self.setup_experiment(info)
                # Now that the process is started, we can configure the circuit
                # (e.g. read/write to tags) and gather the information we need to
                # run the experiment.
                self.configure_experiment()
                # now all the parameters are final, configure data instance
                self.data.get_data_template(self._devices_output_params())
                # store configuration and parameters when the experiment starts
                self.store_info_when_start()
                self._before_start()
                # start thread if configured
                if self._use_default_thread:
                    self._start_thread()
                # run experiment
                self.start_experiment(info)
                # Save the start time in the model as well as H5 file
                self.model.setting.start_time = datetime.datetime.now()
                self.data.set_h5_attrs(start_time=self.model.setting.start_time.strftime(DATETIME_FMT),
                                       start_date=self.model.setting.start_time.strftime(DATE_FMT),
                                       status='started')
                # update experiment history
                self.update_history(start_time=self.model.setting.start_time.strftime(DATETIME_FMT),
                                    status='started')
            else:  # restart experiment
                self.next_trial()

            self.change_state('Running', pausing=False, thread_running=True)

        except Exception as e:
            log.exception(e)
            if self.info and self.info.ui:
                mesg = '''
                    Unable to start the experiment due to an error.  Please correct the
                    error condition and attempt to restart the experiment.  Note that
                    you may have to shut down and start the program again.
                    '''
                log.error(mesg)
                import textwrap
                mesg = textwrap.dedent(mesg).strip().replace('\n', ' ')
                mesg += '\n\nError message: ' + str(e)
                error(mesg)

    def _devices_output_params(self):
        """
        prepare the all device outputs parameters as a orderedDict to be feed to the data class

        Returns:
            orderedDict
        """
        devices_outputs = OrderedDict()
        for kv in sorted(self.devices.keys()):
            devices_outputs[kv] = deepcopy(self.devices[kv]._output_specs.copy())
            # make sure the field required are complete
            device_name = self.devices[kv].name
            if 'dtype' in devices_outputs[kv].keys():
                if 'source' not in devices_outputs[kv].keys():
                    devices_outputs[kv]['source'] = device_name
            # has multiple output
            else:
                for on, op in devices_outputs[kv].items():
                    if 'source' not in op.keys():
                        op['source'] = device_name
                    # if it is a stream, need to update the latest packet count
                    if 'stream' in op.keys():
                        op['stream']['N_packet'] = self.devices[kv].output[on].N_packet
        return devices_outputs

    def _before_start(self):
        """
        Called immediately before starting the experiment (start_experiment() call)
        mostly used for debugging, or some special configuration requires last-minute adjustment
        Returns:
            None
        """
        pass

    def _before_start_validate(self):
        """
        Called during calling start().

        Perform checking and validation before starting the Logic. By default, it checks if all
        devices can be started (i.e. their state is either Ready or Paused), and close the device GUI.
        Override this method if you want to perform other type of checks
        """
        # make sure the subject object has all required information
        if not self.subject:
            log.error('Subject is not defined')
            raise RuntimeError('Subject is not defined')
        if not self.subject.file_path:
            log.error('Subject file not defined. Subject must be in an h5 file')
            raise RuntimeError('Subject file not defined. Subject must be saved in to an h5 file. See '
                               'add_subject_to_h5file method')
        # make sure all devices are in 'Ready' or 'Paused' state, and close any opened device GUI
        for kv, dv in self.devices.items():
            if dv.state not in ('Ready', 'Paused'):
                if dv.state == 'Running':
                    dv.pause()
                else:
                    dv.initialize()
                    if dv.state != 'Ready':
                        raise RuntimeError('The device {} cannot be correctly configured. Check the settings '
                                           'of the device'.format(dv.setting.device_name))
            if dv.info and dv.info.ui:
                dv.info.ui.dispose()

    def store_info_when_start(self):
        """
        store configuration and parameters (experiment and devices) when the experiment starts
        """
        kwfilter = {'group': lambda x: x in ('primary', 'derived')}
        # device settings
        device_settings = {}
        for dn, dv in self.devices.items():
            name = dv.setting.device_name + '_Setting_'
            for dk, dvv in dv.setting.get_parameter_value(kwfilter).items():
                device_settings[name+dk] = dvv
        exp_settings = {}
        for dk, dv in self.setting.get_parameter_value(kwfilter).items():
            exp_settings['Exp_Settings_'+dk] = dv
        self.data.set_h5_attrs(computer=os.environ['COMPUTERNAME'],
                               experimenter=self.experimenter,
                               status='initialized',
                               **exp_settings,
                               **device_settings,  # device settings
                               **self._tosave_para,  # custom info
                               )
        if self._verbose:
            self.data.set_h5_attrs(GlobalSetting=get_config())

        '''
        # This will actually store a pickled copy of the calibration data
        # that can *only* be recovered with Python (and a copy of the
        # Neurobehavior module)
        node._v_attrs['cal_1'] = self.cal_primary
        node._v_attrs['cal_2'] = self.cal_secondary
        '''
        # update experiment history entry in h5 subject file and experiment file
        self.update_history(experiment_name=self.data.store_node_name,
                            experimenter=self.experimenter,
                            computer=os.environ['COMPUTERNAME'],
                            paradigm=self.model.setting.experiment_name,
                            status='initialized',
                            node='/'+self.data.store_node_name,
                            age=self.subject.age,
                            )

    def update_history(self, **kwargs):
        """
        Update experiment history record (in subject file) for current experiment

        Args:
            kwargs: see class subject_history_obj

        Returns:
            None
        """
        self.data.update_experiment_history(**kwargs)

    def stop(self, info=None):
        # stop all devices
        for kv, dv in self.devices.items():
            try:
                dv.pause()
            except AssertionError:
                pass
            dv.stop()
        # stop the Logic
        Logic.stop(self, info)
        # save unsaved data, if the experiment is ran
        # record stopping time (only when experiment has been started)
        if self._started:
            stop_time = datetime.datetime.now()
            self.data.set_h5_attrs(stop_time=stop_time.strftime(DATETIME_FMT),
                                   stop_date=stop_time.strftime(DATE_FMT),
                                   )
            # is the experiment finished?
            if self._complete:
                self.data.set_h5_attrs(status='complete', close_ext_on_finish=True)
                self.update_history(status='complete')
            else:
                self.data.set_h5_attrs(status='aborted', close_ext_on_finish=True)
                self.update_history(status='aborted')
        self.data.save(save_range='all', close_file=True)
        # stop data worker
        self.data.stop_writer()

    # --------------------------------------------------------------------
    # utility methods hidden from GUI
    '''
    Trials are controlled by events. There should be a trial end event which calls stop_trial(),
    as well as an event which calls next_trial()
    '''

    # state variables used to control trial progressing
    _trial_active = Bool(False)
    _pausing      = Bool(False)
    _complete     = Bool(False)  # if the experiment is successfully finished
    _started      = Bool(False)  # if the experiment has been started at least once

    def pause(self, info=None):
        """
        only pause the experiment after trial finish. actual pausing action is performed by
        method pause_experiment
        """
        if self._trial_active:
            self.change_state(pausing=True)
        else:
            self.pause_experiment()

    def pause_experiment(self):
        """
        pause the experiment when pause() is called. To define custom actions in pause() method,
        override the _pause() method
        :return:
        """
        assert self.running(), \
            'Cannot pause {} {}: the Node is not running'. \
                format(self.setting.category, self.name)

        log.info('pausing {} {}...'.
                 format(self.setting.category, self.name))
        # pause all devices !! this should be handled by the _pause() method
        # for kv, dv in self.devices.items():
            # if dv.running():
                # dv.pause()
        self._pause()
        self.change_state('Paused', pausing=False)

        log.info('{} {} was paused'.
                 format(self.setting.category, self.name))

    def log_trial(self, current_trial=None, **kwargs):
        if not current_trial:
            current_trial = self.setting.current_trial
        self.data.log_trial(current_trial, **kwargs)

    def log_event(self, ts, event, current_trial=None):
        if not current_trial:
            current_trial = self.setting.current_trial
        self.data.log_event(ts, event, current_trial)

    def setup_experiment(self, info=None):
        """
        prepare experiment settings, generate data needed to prepare each trial
        this method is called before finalizing Data instance
        Args:
            info: Traitsui Info instance

        Returns:
            None
        """
        raise NotImplementedError

    def configure_experiment(self):
        pass

    def start_experiment(self, info=None):
        # set _started flag
        if not self._started:
            self.change_state(started=True)
        # prepare trial
        self.prepare_trial()
        # start trial
        self.start_trial()

    def start_timer(self):
        pass

    def stop_timer(self):
        pass

    def start_trial_timer(self):
        pass

    def stop_trial_timer(self):
        pass

    def prepare_trial(self):
        # prepare data storage for current trial
        self.data.prepare_trial_storage(self.setting.current_trial)
        self._prepare_trial()

    def _prepare_trial(self):
        pass

    def start_trial(self):
        # start devices (order of starting might matter, e.g. the master device normally should be
        # started at last)
        log.info('*{}.start_trial: starting trial {}'.format(
            self.__class__, self.setting.current_trial))

        self._start_trial()
        self.change_state(trial_active=True)

        # log starting time
        start_time = datetime.datetime.now().strftime(TIME_FMT)
        self.log_trial(trial_start=start_time)

    def _start_trial(self):
        raise NotImplementedError

    def stop_trial(self):
        log.info('*{}.stop_trial: stopping trial {}'.format(
            self.__class__, self.setting.current_trial))

        self._stop_trial()
        # log stopping time
        stop_time = datetime.datetime.now().strftime(TIME_FMT)
        self.log_trial(trial_stop=stop_time)
        self.change_state(trial_active=False)

    def _stop_trial(self):
        raise NotImplementedError

    def next_trial(self):
        # start next trial
        # increase trial count
        self.setting.current_trial += 1
        # prepare trial
        self.prepare_trial()
        # start trial
        self.start_trial()

    def save_data(self, save_range='all', close_file=False):
        self.data.save(save_range, close_file)

    # --------------------------------------------------------------------
    # general purpose event processing
    '''
    two ways to implement event processing:
        1. let event producer (usually different devices) call the process_event method
        2. implement an event queue as well as event processing thread in ExperimentLogic
    
    currently method 1 is used. naively method 2 will have slightly longer delays, but 
    this need to be tested
    '''

    def process_event(self, event):
        """
        general way to process event generated while experiment is running
        :param event: dictionary with name of event as keys, and another dictionary as
        value for event signatures
        :return: None
        """
        for key in event:
            if key == 'trial_stop':
                self.trial_stop_fired()
            else:  # other type of event not defined; implement in subclass
                log.info('Event {} handling is not implemented')

    # process trial_stop event
    def trial_stop_fired(self):
        self.stop_trial()
        if self._pausing:
            self.pause_experiment()
        else:
            if self.setting.current_trial + 1 < self.setting.total_trial:
                time.sleep(self.setting.inter_trial_interval)
                self.next_trial()
            else:
                self.pause()
                self.change_state(complete=True)
                self.stop()

    """
        Traits View (GUI)
    """

    def _get_devices_gui_view(self):
        view_groups = []
        for d in sorted(self.devices.keys()):
            self.add_trait(d, self.devices[d])
            # HGroup(Label(d), Item('object.{}.state'.format(d), style='readonly'), Item("EditButton"))
            vg = self.devices[d].get_gui_viewgroup()
            vg.object = "object.{}".format(d)
            view_groups.append(vg)
        return view_groups

    def default_traits_view(self):

        traits_view = View(
            Item('state', style='readonly', show_label=True),
            VGroup(
                Tabbed(
                    Item('setting', style='custom', show_label=False),
                    VGroup(*self._get_devices_gui_view(), label='Devices'),
                    # Item('data', style='custom', show_label=False),
                    # Item('handler.tracker', style='custom'),
                ),
                Item('handler.toolbar', style='custom', show_label=False),
            ),
            resizable=True,
            title='Experiment: {}'.format(self.name)
        )
        return traits_view
