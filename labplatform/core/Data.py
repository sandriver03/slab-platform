from labplatform.utilities.channel import FileChannel, MemoryChannel
from labplatform.utilities import H5helpers as h5t
# from labplatform.utilities.arraytools import make_dtype
from labplatform.config import get_config
from labplatform.core.Subject import subject_history_obj, Subject
from labplatform.core.Writer import writer_types

import tables as tl
import os
from collections import OrderedDict
from traits.api import List, Any, Int, Event, HasTraits, Bool, Str, Instance, Dict, Enum
import time

import logging
log = logging.getLogger(__name__)


default_writer_params = {'inputs': None,
                         'data_specs': None,
                         'master_file': None,
                         'subject_file': None,
                         'external_file': None,
                         'experiment_name': None,
                         'input_names': None,
                         }


class ExperimentData(HasTraits):
    """
    Store experiment settings and data with pytables (an HDF5 backend)
    currently the h5 file is not handled with thread/subprocess. handling it with threads will likely cause error

    * Currently, the data is separated into 2 parts, both implemented in PyTables:

        1. An overview file, which contains basic information as well as experiment history of each subject

        2. Individual file for each subjects; it does not contain the experiment history (because it can be looked up
        by node attrs), but holds experiment data

    **structure of a data file:**

        -root

            - _v_attrs (tables.AttributesSet)
                number of experiments performed on the subject

            - experiment_history (tables.Table)
                see Core.Subject.subject_history_obj. A record of experiment performed on this subject

            - experiment (tables.Group), title = experimentName_Nr
                _v_attrs (tables.AttributesSet)
                    - start_time             (str)
                    - end_time               (str)
                    - computer               (str)
                    - experimenter           (str)
                    - status                 (str)
                    - Global_setting         (str, all global parameters)
                    - experiment_setting     (str, parameters in model.setting)
                    - device_setting         (str, parameters in device.setting)
                        ...

                data_link (tables.external_link) link to external (actual) data file

                the following data is stored in the external data file:

                _v_attrs (same information as above)

                **trial_0000 (tables.Group), title = trial_0000**
                    _v_attrs (tables.AttributesSet)
                        - trial related settings
                        - start_time
                        - end_time

                    | event_log       (tables.Table)
                    | trial_log       (tables.Table)
                    | data_00         (tables.EArray)
                    | data_01         (tables.EArray)

                    ...
                **trial_0001 (tables.Group), title = trial_0001**
                    ...

            - experiment ...
    """

    '''
    do not reference experiment and device class directly
    need info from experiment class:
        subject
        setting.experiment_name
        setting.current_trial
        setting.trial_duration
        devices[xx]._output_specs
    '''

    # Node to store the data in; should be a tables.node class, the main body of interaction
    store_node = Any(transient=True)
    store_node_name = Str()
    _exp_Nr    = Int()      # current experiment number
    # the data file
    data_file = Any(transient=True)
    # external data file for individual experiment
    ext_data_file = Any(transient=True)
    ext_data_node = Str('/')  # node name to store experiment data in the external file
    ext_data_file_path = Str() # absolute path to the external file
    ext_data_link = Str()  # path to the external link in master file
    # data file path
    data_file_path = Str()
    # node for current subject in the data file
    data_node_name = Str()
    # Node point to the overview file; tables.node, only need to update experiment history
    subject = Instance(Subject)
    subject_node_path = Str(transient=True)
    # the subject file
    subject_file_path = Str(transient=True)
    # the experiment info
    paradigm = Str
    experiment_name = Str
    experiment_current_trial = Int
    experiment_trial_duration = Any
    device_outputs = Instance(OrderedDict)

    # we first hold the data in the RAM (at least) for each trial, then after the trial finishes
    # write the data into hard drive in h5 format
    # need to declare the RAM storage here; or let them be automatically created using initialize()

    # list to store information about data storage in each trial
    data_spec = List
    # data for current trial
    current   = Any
    # data for all trials
    all_trials = List()
    # should data from all trials to be held in RAM?
    _keep_all_data = Bool(False)

    # List of parameters to analyze.  If you set this up properly, then you can
    # re-analyze your data on the fly.
    parameters = List(['to_duration'])

    # what information need to be logged?
    trial_log_updated = Event
    event_log_updated = Event

    performance = Any

    # parameters used to setup writer
    _writer_params = Dict
    writer = Any
    writer_type = Enum('plain', 'thread', 'subprocess')

    # set up HD5 file and node path
    def initialize(self, experiment_name, subject):
        """
        Generate the tables.group node if it is not there, and add meta information to it. Should only be called once in
        one experiment. If experiment is already ran, save old data by calling stop() and start a new experiment

        Args:
            experiment_name: str, name of the experiment paradigm
            subject: subject.Subject instance, which provide the subject information

        Returns:
            None
        """
        self.all_trials = []
        self.data_spec = []
        self._writer_params = dict()
        self._writer_params.update(default_writer_params)

        # check if the subject information is consistent between Data and Experiment instances
        # check subject path
        if self.subject_node_path != '':
            if self.subject_file_path != subject.file_path:
                log.error('Subject file in Data ({}) and Experiment ({}) do not match'.
                            format(self.subject_file_path, subject.file_path))
                raise ValueError('Subject files do not match')
            if self.subject_node_path != subject.node_path:
                log.error('Subject node in Data ({}) and Experiment ({}) do not match'.
                            format(self.subject_node_path, subject.node_path))
                raise ValueError('Subject nodes do not match')
        else:
            self.subject_node_path = subject.node_path
            self.subject_file_path = subject.file_path
        self._writer_params.update({'subject_file': [self.subject_file_path, self.subject_node_path]})

        # check data file path
        if self.data_file_path != '':
            # this only works if full path is specified for both - should be the case in our program
            if self.data_file_path != subject.data_path:
                log.error('Data files in Data ({}) and Experiment ({}) do not match'.format(
                    self.data_file_path, subject.data_path
                    ))
                raise ValueError('Data files in Data {} and Experiment {} do not match'.format(
                    self.data_file_path, subject.data_path
                ))
        else:
            # assign data file
            self.data_file_path = subject.data_path

        data_file = h5t.get_file_handle(self.data_file_path)
        self.data_node_name = subject.str_repr()

        # check if experiment_history table is already created with this data file
        try:
            data_file.get_node('/', self.data_node_name)
        except tl.exceptions.NoSuchNodeError:
            data_file.create_group('/', name=self.data_node_name, title=self.data_node_name)
        try:
            data_file.get_node('/'+self.data_node_name, 'experiment_history')
        except tl.exceptions.NoSuchNodeError:
            # create experiment_history table
            data_file.create_table('/'+self.data_node_name, name='experiment_history', description=subject_history_obj,
                                        title='records of performed experiments on this subject')

        # the name of the node to store experiment data
        data_node = data_file.get_node('/'+self.data_node_name)
        if experiment_name + '_count' in data_node._v_attrs:
            self._exp_Nr = data_node._v_attrs[experiment_name + '_count'] + 1
        else:
            self._exp_Nr = 0
        self.store_node_name = experiment_name + '_' \
            + str(self._exp_Nr).zfill(get_config('FILL_DIGIT'))
        self._writer_params.update({'master_file': [self.data_file_path,
                                                    self.data_node_name,
                                                    self.store_node_name]})

        self.subject = subject
        self.paradigm = experiment_name
        self.experiment_name = self.store_node_name
        self._writer_params.update({'experiment_name': self.store_node_name})
        data_file.close()

    # generate a list to store the data template for each trial
    # TODO: factor in data size change in stream handlers
    def get_data_template(self, device_outputs, ch_type='File'):
        """
        Create prototype trial storage template for the experiment (self.data_spec). By default, it gathers information
        about the data from `experiment.devices[YOURDEVICE]._output_specs`

        The name of the storage will be:

            1. if the device has multiple outputs, then the name will be the 'devicename_key' for each output

            2. otherwise the name will be the name of the device (device.setting.device_name) with metadata:
                | `source`: name of the device providing the data
                | `dtype`:  data type
                | `fs`:     sampling frequency

        Args:
            device_outputs: ordereddict,
            ch_type: type of the storage; 'Ram' for im memory store with MemoryChannel class, 'File' for h5 file store with
             FileChannel class

        Returns:
            None

        """
        self.data_spec = []
        # create store node for the experiment
        data_file = h5t.get_file_handle(self.data_file_path)
        store_node = data_file.create_group('/'+self.data_node_name,
                                            self.store_node_name, title=self.store_node_name)
        # external file for experiment data storage; the file should be in data/FOLDER_NAME/FILE_NAME, with FOLDER_NAME
        # as self.species_self.group_self.name, FILE_NAME as self.store_node_name
        fd_name = self.subject.str_repr()
        df_path = os.path.join(os.path.split(self.data_file_path)[0], fd_name)
        ef_name = self.store_node_name + '.h5'
        self.ext_data_file_path = df_path + os.sep + ef_name
        if not os.path.isdir(df_path):  # create dir if not there
            os.makedirs(df_path)
        data_file.create_external_link(store_node, 'data_link',
                                       df_path + os.sep + ef_name + ':' + self.ext_data_node)
        self.ext_data_link = self.store_node_name + '/' + 'data_link'

        # increase experiment count
        data_node = data_file.get_node('/'+self.data_node_name)
        if self.paradigm + '_count' in data_node._v_attrs:
            data_node._v_attrs[self.paradigm + '_count'] = self._exp_Nr
        else:
            data_node._v_attrs[self.paradigm + '_count'] = 0

        for kv, dv in device_outputs.items():
            self._add_dataChannel(dv, ch_type)
        data_file.close()
        self.device_outputs = device_outputs
        self._writer_params.update({'data_specs': self.data_spec,
                                    'external_file': [self.ext_data_file_path, self.ext_data_node]})
        # name of all outputs, used to check if all data to be saved are finished gathering
        input_names = []
        for outs in self.data_spec:
            input_names.append(outs['name'])
        input_names.extend(['trial_log', 'event_log'])
        self._writer_params.update({'input_names': input_names})
        if self.writer_type == 'thread':
            import queue
            command_q = queue.Queue()
            response_q = queue.Queue()
            self._writer_params['connect_streams'] = True
        elif self.writer_type == 'subprocess':
            import multiprocessing
            command_q = multiprocessing.Queue()
            response_q = multiprocessing.Queue()
            self._writer_params['connect_streams'] = False
        else:
            command_q = None
            response_q = None
            self._writer_params['connect_streams'] = True
        self._writer_params.update({'writer_type': self.writer_type})
        self.writer = writer_types[self.writer_type](command_q, response_q=response_q, **self._writer_params)
        # need to wait some time for the connection of sockets; only relevant when async writers are used
        if self.writer_type in ('subprocess', 'thread'):
            # wait until the subprocess loop is ready, with a timeout
            process_ready = False
            # maximum wait 30s, otherwise stop and raise an error
            if 'timeout' in self._writer_params:
                timeout = self._writer_params['timeout']
            else:
                timeout = get_config("TIMEOUT_PROCESS")
            t0 = time.time()
            self.writer.command_q.put(('get_state', ('_ready', )))
            while not process_ready:
                if time.time() - t0 > timeout:
                    raise TimeoutError('Async DataWriter cannot be started before timeout')
                if not self.writer.response_q.empty():
                    if self.writer.response_q.get()[1]:
                        process_ready = True
                else:
                    time.sleep(0.1)

    def _add_dataChannel(self, ch_specs, ch_type='File'):
        """
        add data channel specification into the data template, i.e. self.data_spec
        Args:
            ch_specs: dictionary of data specifications. format should follow device._output_specs
            ch_type: type of the data storage, 'Ram' or 'File'

        Returns:
            None
        """
        if 'dtype' in ch_specs:  # only one output from the device
            self._add_singleChannel(ch_specs, ch_type)
        else:  # multiple outputs
            for k, do in ch_specs.items():
                if k not in do['source']:
                    self._add_singleChannel(do, ch_type, name_ext=k)
                else:
                    self._add_dataChannel(do, ch_type)

    def _add_singleChannel(self, ch_specs, ch_type='File', name_ext=None):
        """
        add a single data channel specification into the data template, i.e. self.data_spec
        Args:
            ch_specs: dictionary of data specifications. format should follow device._output_specs
            ch_type: type of the data storage, 'Ram' or 'File'
            name_ext: string, name extension for the channel

        Returns:
            None
        """
        metadata_dict = {'source': ch_specs['source'],
                         'dtype': ch_specs['dtype'],
                         'fs': ch_specs['sampling_freq'],
                         'shape': ch_specs['shape']}
        # type of the storage
        if 'channel_type' in ch_specs:
            ch_type = ch_specs['channel_type']
        if ch_type == 'Ram':
            store = MemoryChannel
        elif ch_type == 'File':
            store = FileChannel
        else:
            raise ValueError('storage type {} not defined'.format(ch_type))

        if ch_type == 'Ram' and 'length' in ch_specs and ch_specs['length'] > 0:
            metadata_dict['data_length'] = ch_specs['length']

        if 'name' not in ch_specs:
            if name_ext:
                ch_specs['name'] = ch_specs['source'] + '_' + name_ext
            else:
                ch_specs['name'] = ch_specs['source']

        # TODO
        # if software scaling is specified
        if 'software_scaling' in ch_specs and ch_specs['software_scaling']:
            try:
                ss = ch_specs['software_scaling'].copy()
            except AttributeError:    # not a Sized object
                ss = ch_specs['software_scaling']
            if not isinstance(ss, (list, tuple)):
                # scaling is a number
                ss = list([ss])
            if isinstance(ss, tuple):
                ss = list(ss)
            # scaling the shape parameter, starting from 2nd position; 1st should always be -1
            old_shape = list(metadata_dict['shape'][1:])
            if len(ss) == 1:
                new_shape = [int(old_val/ss[0]) for old_val in old_shape]
                full_scale = ss * len(new_shape)
            elif len(ss) == len(old_shape):
                new_shape = [int(val[0]/val[1]) for val in zip(old_shape, ss)]
                full_scale = ss
            else:
                raise ValueError('Software scaling of {} cannot be done'.format(ss))

            # must be a full size rescaling
            for nd, ns in zip(old_shape, ss):
                if nd % ns:
                    raise ValueError('array shape: {} cannot be fully divided by downscale factor: {}'
                                     .format(old_shape, ss))

            metadata_dict['shape'] = (-1, ) + tuple(new_shape)
            metadata_dict['original_shape'] = (-1, ) + tuple(old_shape)
            metadata_dict['software_scaling'] = ch_specs['software_scaling']
            metadata_dict['sf_full'] = tuple(full_scale)

        stream = None
        if 'stream' in ch_specs:
            stream = ch_specs['stream']
            if 'sf_full' in metadata_dict:
                stream.update({'sf_full': metadata_dict['sf_full']})

        save = True
        if 'save' in ch_specs:
            save = ch_specs['save']

        para_dict = {'name': ch_specs['name'],
                     'type': ch_type,
                     'store': store,
                     'stream': stream,
                     'save': save,
                     'metadata': metadata_dict}
        self.data_spec.append(para_dict)

    # create RAM storage to hold data from different devices
    def prepare_trial_storage(self, current_trial, trial_duration=0):
        """
        Prepare data storage for each trial. `self.data_spec` must already been set.
        """
        self.writer.prepare_trial_storage(current_trial, trial_duration)
        self.experiment_current_trial = current_trial

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
        self.writer.log_event(ts, event, current_trial)

    def log_trial(self, current_trial, **kwargs):
        """
        Add kwargs items into trial log as base strings

        Args:
            current_trial: int, current trial index
            **kwargs: anything to be logged; max length 30

        Returns:
            None
        """

        # This is a very inefficient implementation (appends require
        # reallocating information in memory).
        self.writer.log_trial(current_trial, **kwargs)

    def write(self, key, data, current_trial=None):
        """
        Write data into specific data storage specified by key

        Args:
            key: str, name of the data storage to write to
            data: data to be written
            current_trial: int, current trial number

        Returns:
            None
        """
        if current_trial is None:
            current_trial = self.experiment_current_trial
        self.writer.write(key, data, current_trial)

    def write_to_base(self, data, node_name):
        """
        Write a piece of data into the ext_file.root.node_name. This method is mainly used to write a piece of
        data/ attribute that is too large to fit into the node attributes

        Args:
            data: data to be written
            node_name: str, name of the new node to be created

        Returns:
            None
        """

    def save(self, save_range='current', close_file=False):
        """
        Called by stop_experiment when the stop button is pressed.  This is your chance to save relevant data.

        Note: only MemoryStorage need to be explicitly saved
        """
        self.writer.save(save_range=save_range, close_file=close_file)

    def update_performance(self, trial_log):
        pass

    def update_experiment_history(self, close_on_finish=True, **kwargs):
        """
        Update experiment history record (a table.Table node) in the HDF file. Either add a new entry to the table if
        record about current experiment is not there, or update existing records. Close the file after finish.

        Args:
            close_on_finish: if close the files after finish. default True
            **kwargs: See Core.Subject.subject_history_obj for individual field

        Returns:
            None
        """
        self.writer.update_experiment_history(**kwargs)

    def set_h5_attrs(self, close_on_finish=True, close_ext_on_finish=False, **kwargs):
        """
        Add node attribute set to both the master data file as well as the external file

        Args:
            close_on_finish: if close the master file after finish. default True
            close_ext_on_finish: if close the external file after finish. default False
            **kwargs: values to be added or modified to the node attribute set

        Returns:
            None
        """
        self.writer.set_experiment_attrs(**kwargs)

    def input_finished(self, finished_inputs, current_trial=None):
        """
        signal input(s) has finished for current trial
        Args:
            current_trial: int, trial index
            finished_inputs: list or tuple of str, name of inputs to be set
        Returns:
            None
        """
        if current_trial is None:
            current_trial = self.experiment_current_trial
        self.writer.input_finished(current_trial, finished_inputs)

    def stop_writer(self):
        print('trying to stop data writer...')
        self.writer.stop_writer()

    def close_input(self, inputs_to_close):
        """
        close input stream(s)
        Args:
            inputs_to_close: str or tuple or list; name(s) of the input stream to be closed
        Returns:
            None
        """
        if isinstance(inputs_to_close, str):
            inputs_to_close = (inputs_to_close, )
        for stream_name in inputs_to_close:
            self.writer.disconnect_stream(stream_name)

    def configure_writer(self, **kwargs):
        """
        configure the writer parameters
        :param kwargs: see _writer_params
        :return:
            None
        """
        for k, v in kwargs.items():
            self._writer_params.update({k: v})
