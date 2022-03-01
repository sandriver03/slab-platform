'''
    ----------------------------------------------
    subjects information file
    (do we keep it separate from the data file? probably will save some reading time as the program
    starts. however need to sync this with the information in the data files)
        format: h5 (with pytables module)
        organization:
            -root
                -group (tables.Group)
                    -name (tables.Group)
                        _v_attrs (tables.AttributesSet)
                            -species    (str)
                            -birth      (str)(converts to datetime.date after reading)
                            -age        (float)
                            -age_unit   (str)
                            -sex        (set)
                            -genotype   (str)
                            -phenotype  (str)
                            -note       (str)
                        experiments (tables.Table)
                            record datetime, paradigm, experimenter, computer of each experiment performed
                            on the subject; should be added whenever a paradigm is started
                ! do we need a subfolder at level of species?
    ------------------------------------------------
'''

from labplatform.config import get_config
from labplatform.utilities import H5helpers as h5t

import tables as tl
import datetime, os
from traits.api import HasTraits, Str, Float, Enum, Instance, List, Button, \
    Any, Property, cached_property, Date, TraitError
from traitsui.api import Handler, Action, View, Group, Item, TableEditor,\
    UItem, ObjectColumn, HGroup, VSplit, DateEditor
from traitsui.menu import OKButton, CancelButton
from traitsui.table_filter import MenuFilterTemplate, EvalTableFilter
from pint import UnitRegistry


import logging
log = logging.getLogger(__name__)

date_fmt = get_config('DATE_FMT')
tl_sep = get_config('PYTABLES_SEP')


class Subject(HasTraits):
    """
    Defines the following attributes for each subject:

        | **name**
        | **group**
        | **species**
        | **birth**
        | **genotype**
        | **phenotype**
        | **note**
        | age
        | file_path
        | node_path
        | experiments
        | data_path
        | node
        | cohort

    The attributes `name`, `group` and `species` must be specified. `birth`, `genotype`, `phenotype` and `note` can be
    left blank. Other attributes are Property traits and cannot be set.
    """

    name         = Str('', attr=True,
                       dsec='Once added, the subject\'s name, group and species cannot be modified')
    age_unit     = Enum(['day', 'month', 'year'], attr=True)
    age          = Property(depends_on=['age_unit', 'birth'], save=False)
    birth        = Date(datetime.date(1990, 1, 1), dsec='Month-Day-Year', attr=True)
    species      = Enum(['Human', 'Mouse'], attr=True,
                        dsec='Once added, the subject\'s name, group and species cannot be modified')
    sex          = Enum(['M', 'F'], attr=True)
    group        = Str('', attr=True, dsec='The experimental group(s) the subject belongs to')
    genotype     = Str('WT', attr=True)
    phenotype    = Str('', attr=True)
    note         = Str('', attr=True)
    cohort       = Str('', attr=True, dsec='experiment group name')
    data_path    = Property(Any, depends_on=['name', 'group', 'species'], save=False)
    rec_name     = Str('experiment_history', attr=True)
    parent_model = Any(None, save=False)
    file_path    = Str(attr=True)  # path to h5 file containing subjects information
    node         = Instance(tl.Group)  # pytables Group object
    node_path    = Str(attr=True)  # path in the h5 file to the node
    experiments  = List(Str, attr=True)       # the experiments the subject has participated
    # check if the subject is unique
    _valid_sub    = Property(depends_on=['name', 'group'], save=False)

    # ---------------------------------------------------------------------------
    # traits related methods
    # ---------------------------------------------------------------------------

    '''
    def _age_unit_changed(self, old_unit, new_unit):
        ur = UnitRegistry()
        age_dm = self.age * ur(old_unit)
        self.age = age_dm.to(new_unit)._magnitude
    '''

    def _species_changed(self, old_species, new_species):
        if new_species == 'Human':
            self.age_unit = 'year'
        elif new_species == 'Mouse':
            self.age_unit = 'day'
        else:
            raise ValueError('Species: {} not found.'.format(self.species))

    def _get__valid_sub(self):
        if self.parent_model.edit_mode == 'add':
            return self.name != '' and \
                [self.group, self.name] not in self.parent_model.unique_subs
        elif self.parent_model.edit_mode == 'edit':
            return True
        else:
            raise ValueError('Edit mode: {} is not known'.format(self.parent_model.edit_mode))

    def _get_age(self):
        delta_days = (datetime.date.today() - self.birth).days
        if self.age_unit == 'day':
            return delta_days
        else:
            ur = UnitRegistry()
            age_dm = delta_days * ur('day')
            return age_dm.to(self.age_unit)._magnitude

    def _get_data_path(self):
        # set to datafolder/file_name/species_group_name.h5
        # file_name is the file in which this subject is stored
        if not self.file_path:
            return None
        else:
            file_name = os.path.splitext(os.path.basename(self.file_path))[0]
            # return os.path.join(get_config('DATA_ROOT'),
            #                    file_name,
            #                    self.species+'_'+self.group+'_'+self.name+'.h5')
            return os.path.join(get_config('DATA_ROOT'),
                            file_name,
                            file_name + '.h5')

    def str_repr(self):
        return self.cohort + '_' + self.group + '_' + self.species + '_'+self.name

    def _age_unit_default(self):
        if self.species == 'Human':
            return 'year'
        elif self.species == 'Mouse':
            return 'day'
        else:
            raise ValueError('Unit for Species: {} is not defined.'.format(self.species))

    def get_dateEditor(self):
        return DateEditor(allow_future=False, strftime=date_fmt)

    def __str__(self):
        return 'Subject: species {}, name {}, sex {}, group {}, cohort {}'.format(
            self.species, self.name, self.sex, self.group, self.cohort)

    def __repr__(self):
        txt = super(Subject, self).__repr__()
        return txt + ': species {}, name {}, sex {}, group {}, cohort {}'.format(
            self.species, self.name, self.sex, self.group, self.cohort)

    def default_traits_view(self):
        #print(self.parent_model.unique_subs)
        #print(self.valid_sub)
        return View(
            Group(
                Item(name='cohort', style='readonly', tooltip=self.trait('cohort').dsec),
                Item(name='species', enabled_when="parent_model.edit_mode != 'edit'",
                     tooltip=self.trait('species').dsec),
                Item(name='group', tooltip=self.trait('group').dsec,
                     enabled_when="parent_model.edit_mode != 'edit'"),
                Item(name='name', enabled_when="parent_model.edit_mode != 'edit'",
                     tooltip=self.trait('name').dsec),
                Item(name='sex'),
                Item(name='birth', editor=self.get_dateEditor(), tooltip=self.trait('birth').dsec),
                # Item(name='age_unit'),
                # Item(name='age', format_str='%.1f'),
                Item(name='genotype'),
                Item(name='phenotype'),
                Item(name='experiments', style='readonly'),
                Item(name='note'),
                show_border=True,
                label='subject information',
                        ),
            kind='modal',
            buttons=[Action(name='OK', enabled_when='_valid_sub'), CancelButton],
            title='Edit subject'
        )

    # ---------------------------------------------------------------------------
    # h5 file related methods
    # ---------------------------------------------------------------------------

    def update_info_to_h5file(self, file=None, close_on_finish=True):
        """
        write subject related information into an h5 node's attributes
        :param file: an h5 file handle or a path to the file to write into
        :param close_on_finish: if true close the file after method execution
        :return:
        """
        file = self.find_file(file)
        sub_node = self.find_node(file)
        log.debug('update subject: {} information on node: {} in file: {}'.
                  format(self.group+tl_sep+self.name, sub_node._v_pathname, file))
        for trs in self.traits(attr=True).keys():
            if trs != 'birth':
                sub_node._f_setattr(trs, self.__getattribute__(trs))
            else:
                ds = self.__getattribute__(trs)
                ds = ds.strftime(date_fmt)
                sub_node._f_setattr(trs, ds)
        # commit changes
        sub_node._f_flush()
        # close if input is a path to file
        if close_on_finish:
            sub_node._v_file.close()

    def read_info_from_h5file(self, file=None, close_on_finish=True):
        """
        read attributes from .h5 file
        :param file: an h5 file handle or a path to the file to read from
        :param close_on_finish: if true close the file after method execution
        :return:
        """
        try:
            file = self.find_file(file)
            sub_node = self.find_node(file)
            log.debug('Reading from file {}, subject: {} in group: {}'
                      .format(sub_node._v_file.filename, self.name, self.group))
            for trs in sub_node._v_attrs._f_list('user'):
                if trs == 'birth':
                    ds = datetime.datetime.strptime(sub_node._v_attrs.__getattr__(trs), date_fmt)
                    self.__setattr__(trs, ds.date())
                else:
                    try:
                        self.__setattr__(trs, sub_node._v_attrs.__getattr__(trs))
                    except TraitError:  # the program is trying to set Property traits
                        pass
            # close if input is a path to file
            if close_on_finish:
                sub_node._v_file.close()
        except tl.NoSuchNodeError:
            log.error('subject {} in group {} not found!'.format(self.name, self.group))

    def add_subject_to_h5file(self, file=None, close_on_finish=True):
        """
        add new subject into the subject file (a .h5 file). if no existing file is found, create
        a new file in default subject file location, use provide file argument as file name
        subject's name follows species_group_name
        :param file: an h5 file handle or a path to the file to write into
        :param close_on_finish: if true close the file after method execution
        :return:
        """
        # support relative/absolute path for file argument. when relative, save to SUBJECT_ROOT directory
        if file is not None:
            if isinstance(file, str) and not os.path.isabs(file):
                file = os.path.join(get_config('SUBJECT_ROOT'), file)

        file = self.find_file(file)
        if isinstance(file, str):  # assume the file is closed
            file_handle = h5t.get_handler_by_name(file)
            # file_handle = tl.file._open_files.get_handlers_by_name(file)
            if file_handle:
                h = file_handle
            else:
                suffix = ''
                if not file.lower().endswith('.h5'):
                    suffix = '.h5'
                name = os.path.join(get_config('SUBJECT_ROOT'), (file + suffix))
                h = tl.open_file(name, mode='a')
        elif isinstance(file, tl.File):
            h = file
        elif isinstance(file, tl.Node):
            h = file._v_file
        else:
            raise ValueError('provided file: {} not recognized'.format(file))
        # first check if the group already exist
        try:
            group_node = getattr(h.root, self.group)
            try:
                getattr(group_node, self.name)
                raise ValueError('The subject: {} in group: {} already exist!'
                                 .format(self.name, self.group))
            except tl.NoSuchNodeError:
                group_node = group_node
        except tl.NoSuchNodeError:
            # create the node for the group
            log.debug('create group: {}'.format(self.group))
            group_node = h5t.append_node(h.root, self.group)

        # create node for the subject
        log.debug('create subject: {} under group: {}, in file: {}'.
                  format(self.name, self.group, group_node._v_file.filename))
        sub_node = h5t.append_node(group_node, self.name)
        self.node_path = sub_node._v_pathname
        # writes information into the node attributes
        for trs in self.traits(attr=True).keys():
            if trs != 'birth':
                sub_node._f_setattr(trs, self.__getattribute__(trs))
            else:
                ds = self.__getattribute__(trs)
                ds = ds.strftime(date_fmt)
                sub_node._f_setattr(trs, ds)
        # create table to store experiment history
        h.create_table(sub_node, name=self.rec_name, description=subject_history_obj,
                       title='records of performed experiments on this subject')
        # commit changes
        sub_node._f_flush()
        self.node = sub_node
        if close_on_finish:
            sub_node._v_file.close()

    def write_history_to_h5fle(self, data, file=None, close_on_finish=True):
        """
        write experiment history into an h5 node's experiments table
        :param file: an h5 file handle or a path to the file to write into
        :param data: dictionary contains 'start_time', 'paradigm', 'experimenter', 'status',
                'node' and 'computer' as keys
        :param close_on_finish: if true close the file after method execution
        :return:
        """
        file = self.find_file(file)
        sub_node = self.find_node(file)
        log.debug('writing experiment history to node {} in file {}'.format(sub_node._v_pathname, file))
        exp_table = sub_node.__getattr__(self.rec_name).row
        for key in data.keys():
            exp_table[key] = data[key]
        # commit changes
        exp_table._flush_buffered_rows()
        # close if input is a path to file
        if close_on_finish:
            sub_node._v_file.close()

    def find_file(self, file=None):
        if file is None:
            if self.parent_model is not None:  # parent_model should be a subjectList class
                if self.parent_model.file_handle and self.parent_model.file_handle.isopen:
                    file = self.parent_model.file_handle  # handle to tables.File
                else:
                    file = self.parent_model.file_path  # string, file path
            elif self.node is not None and self.node._v_isopen:
                file = self.node._v_file  # handle to tables.File
            else:
                file = self.file_path  # string, file path
        self.file_path = file.filename if isinstance(file, tl.File) else file
        return file

    def find_node(self, file=None):
        """
        returns the node the subject should be linked
        :param file: an h5 file handle or a path to the file to write into
        :return: tables.Group object
        """
        if file is None:
            file = self.find_file()

        if self.group == '' or self.name == '':
            raise ValueError('the name or group of the subject is not specified')
        else:
            node = os.path.join(os.sep, self.group, self.name)
            node = node.replace(os.path.sep, tl_sep)
            if isinstance(file, str):
                # get file handle
                if self.node is not None and self.node._v_isopen and file == self.node._v_file.filename:
                    # the object already associated with a file, which is the same as passed in file
                    h = self.node._v_file
                else:  # assign to new file
                    # check if the passed in file is opened
                    file_handle = h5t.get_handler_by_name(file)
                    # file_handle = tl.file._open_files.get_handlers_by_name(file)
                    if file_handle:
                        h = file_handle
                    else:
                        h = tl.open_file(file, mode='a')
                # get node handle
                try:
                    sub_node = h.get_node(node, classname='Group')
                except tl.NoSuchNodeError:
                    msg = 'node :{} in file: {} not found'.format(node, file)
                    log.debug(msg)
                    if self.node is None:
                        h.close()
                    raise ValueError(msg)
            elif isinstance(file, tl.File):
                sub_node = file.get_node(node, classname='Group')
            elif isinstance(file, tl.Node):
                sub_node = file
                if file._v_pathname != node:
                    msg = 'the node provided {} is not at the default node path {}'\
                        .format(file._v_pathname, node)
                    log.warning(msg)
            else:
                raise ValueError('provided file: {} not recognized'.format(file))
            self.node = sub_node
            return sub_node

    def default_nodepath(self):
        """
        Get default path of the subject in the HDF file. It is in form of '/self.group/self.name'

        Returns:
            str, path to the node in the HDF5 file
        """
        return tl_sep + self.group + tl_sep + self.name

    def list_experiment(self, paradigm=None, status='complete', cond_str=None):
        """
        search and list experiments of the subject
        Args:
            paradigm: string, experiment to be searched; if none list all experiment paradigms
            status: string, status of the experiment
            cond_str: custom conditions; override paradigm and status arguments. see table.where()
        Returns:
            tuple of strings
        """
        if cond_str:
            cond = cond_str
        else:
            cond = ""
            if paradigm is not None:
                cond += "(paradigm == '{}')".format(paradigm)
            if status not in (None, 'all'):
                if cond:
                    cond += " & (status == '{}')".format(status)
                else:
                    cond += "(status == '{}')".format(status)
        nh = self.find_node()
        # list experiments; order should be paradigm, experiment_name, start_time, status
        re = tuple([(r['paradigm'], r['experiment_name'], r['start_time'], r['status'])
                    for r in nh.experiment_history.where(cond)])
        # close file
        nh._v_file.close()
        return re

    def load_data(self, experiment):
        """
        load data for a particular experiment
        Args:
            experiment: string, experiment name
        Returns:
            tables.file handle
        """
        fname_str ="{}_{}_{}_{}".format(self.cohort, self.group, self.species, self.name)
        folder_name = os.path.join(os.path.dirname(self.data_path), fname_str)
        exp_str = experiment + '.h5'
        return tl.open_file(os.path.join(folder_name, exp_str), mode='r')


default_subject_file = os.path.join(get_config('SUBJECT_ROOT'), 'Test_0.h5')


class SubjectList(HasTraits):
    subjects         = List(Instance(Subject))
    cohort           = Str()
    Add              = Button
    Edit             = Button
    selected_subject = Instance(Subject)
    unique_subs      = Property(depends_on='subjects')
    edit_mode        = Enum('add', 'edit')
    parent_model     = Any(None)
    file_path        = Str()  # path to h5 file containing subjects information
    file_handle      = Instance(tl.File)
    # ---------------------------------------------------------------------------
    # traits related methods
    # ---------------------------------------------------------------------------

    def _file_path_default(self):
        fname = self.cohort + '.h5'
        path = os.path.join(get_config('SUBJECT_ROOT'), fname)
        # check if path matches
        if self.parent_model:  # parent model should be experimentlogic class?
            if path != self.parent_model._settings['Subject_File_Path']:
                log.error('subject file path not match: subjectlist {} vs. GUI {}'.
                          format(path, self.parent_model._settings['Subject_File_Path']))
                return None
        return path

    # property unique_subs
    @cached_property
    def _get_unique_subs(self):
        return [[sub.group, sub.name] for sub in self.subjects]

    # subject_list editor
    def get_default_editor(self):
        return TableEditor(
            editable=False,
            sortable=True,
            # auto_size = True,
            columns=[ObjectColumn(name='species', width=0.1),
                     ObjectColumn(name='name', width=0.1),
                     ObjectColumn(name='age', width=0.1, format='%.1f', tooltip='age unit: Mouse=day, Human=year',
                                  horizontal_alignment='center'),
                     ObjectColumn(name='sex', width=0.1),
                     ObjectColumn(name='group', width=0.1),
                 ],
            selected='selected_subject',
            filters=[MenuFilterTemplate],
            search=EvalTableFilter(),
            show_toolbar=True,
            row_factory=Subject
            )

    def add_subject(self, new_subject=None):
        self.edit_mode = 'add'
        if new_subject is None:
            new_subject = Subject(cohort=self.cohort)
            new_subject.parent_model = self
            user_action = new_subject.edit_traits()
            if user_action.result:
                self.subjects.append(new_subject)
                new_subject.add_subject_to_h5file(self.file_handle)
        else:
            if self.validate_added_subject(new_subject):
                if not new_subject.cohort:
                    new_subject.cohort = self.cohort
                elif new_subject.cohort != self.cohort:
                    log.warning('the cohort name of the new subject')
                self.subjects.append(new_subject)
                new_subject.add_subject_to_h5file(self.find_file())
            else:
                raise ValueError('subject with the same name already exists!')
        # return a handle to the new subject
        return new_subject

    def validate_added_subject(self, new_subject):
        return [new_subject.group, new_subject.name] not in self.unique_subs

    def edit_subject(self):
        self.edit_mode = 'edit'
        self.selected_subject.parent_model = self
        # print('Editing subject {} information'.format(self.selected_subject.name))
        user_action = self.selected_subject.edit_traits()
        if user_action.result:
            self.selected_subject.update_info_to_h5file(self.file_handle)
        # return a handle to the edited subject
        return self.selected_subject

    def _Add_fired(self):
        self.add_subject()

    def _Edit_fired(self):
        self.edit_subject()

    def default_traits_view(self):
        return View(VSplit(
                Group(
                        Item(name='subjects', editor=self.get_default_editor(), show_label=False, height=400),
                        show_border=True,
                        ),
                HGroup(
                        Item(name='Add', show_label=False),
                        Item(name='Edit', show_label=False, enabled_when="selected_subject"),
                    ),
            ),
            # buttons=[OKButton],
            title='Subjects Information',
            kind='live',
            resizable=True,
        )

    # ---------------------------------------------------------------------------
    # h5 file related methods
    # ---------------------------------------------------------------------------

    def read_from_h5file(self, file=None, close_file_on_finish=True):
        """
        # read subjects from h5file into self.subjects
        all subject nodes (pytable.Group) have depth of 2
        :param file: an h5 file handle or a path to the file containing the subject information
        :return: if the file is opened
        """
        h = self.find_file(file)

        log.debug('reading subject information from file: {}'.format(h.filename))
        to_append = False
        for node in h.walk_nodes(h.root, 'Group'):
            if node._v_depth == 2:  # a subject group
                # check if the subject already exists in the list
                name = [node._v_parent._v_name, node._v_name]
                if name not in self.unique_subs:
                    to_append = True
                    sub = Subject(group=node._v_parent._v_name, name=node._v_name)
                else:
                    sub = self.subjects[self.unique_subs.index(name)]

                sub.read_info_from_h5file(h, close_on_finish=False)
                # check if the cohort name matches
                if sub.cohort != self.cohort:
                    log.error('subject cohort name: {} does not match group cohort name: {}'.
                              format(sub.cohort, self.cohort))

                if to_append:
                    self.subjects.append(sub)

        if close_file_on_finish:
            h.close()
        else:
            self.file_handle = h

    def find_file(self, file=None):
        """
        Get the file handle to the file the instance is associated with.

        If `file` is not provided, it uses parameter `file_path`. Error is raised if `file_path` is None. If a str or
        any class of pytables is provided, it compares the str/pytables file path with `file_path`. If they are not the
        same an error is raised.

        Args:
            file: str, name of the file this instance should be linked to

        Returns:
            tables.File
        """
        if file is None:
            file = self.file_path

        if isinstance(file, str):
            file_handle = h5t.get_handler_by_name(file)
            # fh = tl.file._open_files.get_handlers_by_name(file)
            if file_handle:
                h = file_handle
            else:
                h = tl.open_file(file, mode='a')
        elif isinstance(file, tl.File):
            h = file
        elif isinstance(file, tl.Node):
            h = file._v_file
        else:
            raise ValueError('Unknown input type: {}'.format(file.__class__))

        if self.file_path:
            if h.filename != self.file_path:
                raise ValueError('Provided file name: {} does not match stored file name: {}'
                                 .format(h.filename, self.file_path))

        return h


class subject_history_obj(tl.IsDescription):
    """
    defines which fields are present in experiment history record, as a pytable.table object
    """
    experiment_name   = tl.StringCol(20)
    start_time        = tl.StringCol(24)
    paradigm          = tl.StringCol(20)
    experimenter      = tl.StringCol(20)
    computer          = tl.StringCol(20)
    status            = tl.StringCol(16)    # if the experiment is finished or aborted
    node              = tl.StringCol(30)    # path of the node in the data file
    age               = tl.Float32Col()     # age of the subject when the experiment is performed


ATTRS_ORDER = ['age', 'computer', 'experiment_name', 'experimenter', 'node', 'paradigm', 'start_time', 'status']


def get_cohort_names():
    """
    get the name of all existing cohorts
    Returns:
        list of str
    """
    subject_path = get_config('SUBJECT_ROOT')
    return [fn.strip('.h5') for fn in os.listdir(subject_path) if fn.endswith('.h5')]


def create_cohort(cohort_name, ):
    """
    create a new cohort. raise error if the name already exist
    Args:
        cohort_name: str, name of the cohort to create
    Returns:
        a SubjectList class instance
    """
    # check if the group already exist
    existing_cohorts = get_cohort_names()
    if cohort_name in existing_cohorts:
        raise ValueError('subject group: () already exists!'.format(cohort_name))
    # create a new cohort
    sl = SubjectList(cohort=cohort_name)
    # create file in hard drive
    fh = sl.find_file()
    fh.close()
    return sl


def load_cohort(cohort_name):
    """
    load an existing cohort. raise error if the name does not exist
    Args:
        cohort_name: str, name of the cohort to load
    Returns:
        a SubjectList class instance
    """
    existing_cohorts = get_cohort_names()
    if cohort_name in existing_cohorts:
        fname = os.path.join(get_config('SUBJECT_ROOT'), cohort_name + '.h5')
        sl = SubjectList(cohort=cohort_name)
        sl.read_from_h5file(fname)
    else:
        raise ValueError('subject group: () does not exist!'.format(cohort_name))
    return sl


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

    h5path = os.path.join(get_config('SUBJECT_ROOT'), 'Test_0.h5')
    sl = SubjectList(file_path=h5path)
    sl.read_from_h5file()
    # sl.configure_traits()
