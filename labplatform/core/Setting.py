"""
    hold parameters used for the experiment
    controller should get the parameters from this class to configure the experiment

    parameters are defined with traits package
    each parameter should have the following metadata attributes:
        editable: boolean, if this parameter can be edited through GUI; ignored for derived parameters
        group:    string with value 'primary', 'derived' or 'status'; only primary parameters could be
                  editable. derived parameters are those intermediate parameters calculated from primaries.
                  status parameters are used to indicate experiment status and are not editable
                  value could also be 'ignored', indicating the parameter is not used
        dsec:     string describing what this parameter is (please specify unit!)
        context:  bool, used to indicate which parameter change triggers action in program. Primary
                  parameters are automatically given this metadata with value True.
        noshow:   bool, if the parameter is displayed in the GUI. by default is True.
        position: tuple, position of the parameter in the GUI. if not provided then use sorted() to get the order.
                  *important* index starts with 1 instead of 0
        reinit:   bool, if changing this parameter requires re-initialization of the Logic afterwards. If True, the
                  parameter cannot be changed while the Logic is running
        dynamic:  bool, a dynamic parameter can be changed in any logic state. by setting this to False, the parameter
                  cannot be modified when the Logic is running
"""

from labplatform.config import get_config

from traits.api import HasTraits, Int, Float, Property, Instance, Dict, \
    List, Str, Any, HasStrictTraits, Enum, CInt, CFloat, cached_property
from traitsui.api import View, Item, VGroup, HGroup, Include, HSplit
import datetime


PARAMETER_FILTER = {
        'type':     lambda x: x not in ('event', 'python'),
        'group':    lambda x: x not in ('ignored', 'derived', 'status'),
        }

TraitsVal_default = (0, '', b'', u'', False, 0+0j)

primary_setting = {}
derived_setting = {'style': 'readonly'}
status_setting  = {'style': 'readonly'}
default_param_status = ['current_trial', 'total_trial', 'start_time', 'total_duration', 'remain_time']


def verify_para_from_global(local_para, global_para):
    """
    load default values from global setting for those defined in globals; values defined
    locally are preserved
    :param local_para: dict, see parameter structure
    :param global_para: dict
    :return: dict
    """

    if 'metadata' in local_para:
        for pkey in global_para['metadata'].keys() - local_para['metadata'].keys():
            local_para['metadata'][pkey] = global_para['metadata'][pkey]
    for pkey in global_para.keys() - local_para.keys():
        local_para[pkey] = global_para[pkey]

    return local_para


class Setting(HasStrictTraits):
    """
    Contain all configurable parameters for the Logic. Only parameters defined here can be interacted with using
    `configure` method from Logic

    * Parameters are defined with traits package. Each parameter should have the following metadata attributes:
        *editable*: boolean, if this parameter can be edited; default is True for primaries and False for others

        *group*: string with value 'primary', 'derived' or 'status'; only primary parameters could be editable.
        `derived` parameters are those intermediate parameters calculated from primaries. `status` parameters are used
        to indicate experiment status and are not editable. if no value is given, by default set to `primary`

        *dsec*: string describing what this parameter is (please specify unit!)

        *context*:  bool, used to indicate if the parameter change need to be applied to take effect. Primary
                  parameters are automatically given this metadata with value True.
        *reinit*:   bool, used to indicate if the parameter change requires resetting the logic. Primary
                  parameters are automatically given this metadata with value True.
        *dynamic*:  bool, a dynamic parameter can be changed in any logic state. by setting this to False, the
                  parameter cannot be modified when the Logic is running
        noshow:   bool, if the parameter is displayed in the GUI. by default is True.
    """

    _global_paras_lookup = Dict(group='ignored')

    # keep track of the name of parameters
    _para_list = List(group='ignored')

    # variable definition
    category = Str(group='status', dsec="type of 'device' or 'experiment'", noshow=True)

    # common variables
    control_interval = Float(get_config('CONTROL_INTERVAL'), group='primary', dsec='interval at which the logic'
                            'is checking its state. (s)', reinit=False)

    # for testing
    def __para_list_default(self):
        return self.trait_names(**PARAMETER_FILTER)

    def __global_paras_lookup_default(self):
        return {}

    def __init__(self, **kwargs):
        HasTraits.__init__(self, **kwargs)
        # first check if any global parameter need to be imported
        # append to _para_list from global settings when the parameter is not defined locally
        for key in (self._global_paras_lookup.keys() - set(self._para_list)):
            self.__class__.trait_from_dict(key, get_config(self._global_paras_lookup[key]))
            self._para_list.append(key)
        # load default values from global setting for those defined in globals; values defined
        # locally are preserved
        for key in (self._global_paras_lookup.keys() & set(self._para_list)):
            self.__class__.trait_from_dict(key, get_config(self._global_paras_lookup[key]))
        # setup metadata tag for trait notifier
        for key in self._para_list:
            if not self.trait(key).group:
                self.trait(key).group = 'primary'
            if self.trait(key).group == 'primary':
                if 'context' not in self.trait(key).__dict__:
                    self.trait(key).context = True
                if 'reinit' not in self.trait(key).__dict__:
                    self.trait(key).reinit = True
                # for dynamic property, reinit = True -> dynamic = False
                if self.trait(key).reinit:
                    self.trait(key).dynamic = False
                else:
                    # by default, if a parameter does not require reinit, then it is dynamic
                    # however, certain parameters, e.g. for the FILR camera, can only be modified when not running
                    if 'dynamic' not in self.trait(key).__dict__:
                        self.trait(key).dynamic = True

    @classmethod
    def trait_from_dict(cls, trait_name, item):
        """
        *!!! creating new trait is buggy in subclasses; only set meta-parameters for now*

        Create a class trait or set metadata/default value based on a dict; each dictionary should have the following
        keys:

           *name*: str, name of the parameter

           *type*: trait type, defined in Trait package

           *default_val*: default value for the parameter; if not provided then use trait defaults

           *metadata*: a dictionary, containing the following keys:

              editable: boolean, if the parameter can be set with GUI; can be omitted if group is not primary

              group: string with value 'primary', 'derived' or 'status'; only primary parameters could be editable.
              derived parameters are those intermediate parameters calculated from primaries. status parameters are used
              to indicate experiment status and are not editable value could also be 'ignored', indicating the
              parameter is not visible

              dsec: string describing what this parameter is (please specify unit!); if not provided then use
              'description missing'

           *context*:  bool, used to indicate which parameter change triggers action in program. Primary
               parameters are automatically given this metadata with value True.

           *noshow*:   bool, if the parameter is displayed in the GUI. by default is True.

           *position*: tuple, position of the parameter in the GUI. if not provided then use sorted() to get the order.
               *important* index starts with 1 instead of 0

        """

        if trait_name not in cls.class_trait_names():
            # raise ValueError('trait with name: {} already exist!'.format(trait_name))
            # create trait
            if 'dsec' not in item['metadata']:
                item['metadata']['dsec'] = 'description missing'
            if 'default_val' in item:
                item['default_value'] = item['default_val']
            cls.add_class_trait(trait_name, item['type'](**item['metadata']))
        else:
            # take missing value from default
            for pkey in item['metadata'].keys() - cls.class_traits()[trait_name].__dict__.keys():
                cls.class_traits()[trait_name].__dict__[pkey] = item['metadata'][pkey]
            if 'default_val' in item and cls.class_traits()[trait_name].default in TraitsVal_default:
                # print('assign default value: {} to trait: {}'.format(item['default_val'], trait_name))
                cls.class_traits()[trait_name].default_value(
                    cls.class_traits()[trait_name].default_value()[0], item['default_val'])

    @classmethod
    def get_parameters(cls, kwfilter=PARAMETER_FILTER):
        '''
        by default, get primary parameters; a dictionary defining kw:value can be set as filter to choose
        different group of parameters
        :param kwfilter: dictionary
        :return:
        '''
        return sorted(cls.class_trait_names(**kwfilter))

    @classmethod
    def get_parameter_info(cls, kwfilter=PARAMETER_FILTER):
        '''
        Dictionary of available parameters and their corresponding human-readable label

        By default, get primary parameters; a dictionary defining kw:value can be set as filter to choose
        different group of parameters
        '''
        traits = cls.class_traits(**kwfilter)
        return dict((name, trait.dsec) for name, trait in traits.items())

    @classmethod
    def get_parameter_label(cls, parameter):
        return cls.get_parameter_info()[parameter]

    @classmethod
    def get_invalid_parameters(cls, parameters):
        '''
        Return undefined, non-primary parameters

            :param parameters: a list of strings to be checked

            :return: list
        '''
        return [p for p in parameters if p not in cls.get_parameters()]

    @classmethod
    def pp_parameters(cls):
        '''
        Utility classmethod for pretty-printing the list of parameters to the command line.
        '''
        par_info = cls.get_parameter_info()
        parameters = sorted(par_info)
        parameters = [(key, par_info[key]) for key in parameters]

        # Add the column headings
        parameters.insert(0, ('Variable Name', 'Description'))
        parameters.insert(1, ('-------------', '-----------'))

        # Determine the padding we need for the columns
        col_paddings = []
        for i in range(len(parameters[0])):
            sizes = [len(row[i]) if row[i] != None else 0 for row in parameters]
            col_paddings.append(max(sizes))

        # Pretty print the list
        print('\n')
        for i, row in enumerate(parameters):
            if row[1] is None:
                print(' '*4 + row[0].ljust(col_paddings[0]+5) + ' ')
            else:
                print(' '*4 + row[0].ljust(col_paddings[0]+5) + row[1])

    # get a dictionary of parameters with value
    def get_parameter_value(self, kwfilter=PARAMETER_FILTER):
        '''
        Dictionary of available parameters and their corresponding value

        By default, get primary parameters; a dictionary defining kw:value can be set as filter to choose
        different group of parameters
        '''
        traits = self.trait_names(**kwfilter)
        return dict((name, getattr(self, name)) for name in traits)

    # get default view for the class
    def default_traits_view(self):
        '''
        default structure of the view:
        [HGroup(status parameters)
        HSplit[VGroup(primary parameters)| VGroup(derived parameters)]]
        '''
        kwset = {'show_border': True}
        # status group
        status_group = self.getViewGroup(group_name='status', align='h', setting_group=kwset,
                                         setting_item=status_setting)
        # primary group
        primary_group = self.getViewGroup(group_name='primary', align='v', setting_group=kwset,
                                          setting_item=primary_setting)
        # derived group
        derived_group = self.getViewGroup(group_name='derived', align='v', setting_group=kwset,
                                          setting_item=derived_setting)

        # construct view from the 3 groups
        return View(VGroup(status_group,
                           HSplit(primary_group, derived_group),
                           ),
                    resizable=True,
                    )

    def get_view_itemlist(self, group_name, **kwargs):
        """
        generate the view content group
        :param group_name: str
        :param setting: dict
        :return: a list, used to set the 'content' of a view group object
        """
        trs = sorted(self.trait_names(group=group_name))
        content = []
        names = []
        pos = []
        for item in trs:
            if 'noshow' in self.trait(item).__dict__ and self.trait(item).noshow:
                continue
            view_block = Item(item, tooltip=self.trait(item).dsec, **kwargs)
            if self.trait(item).group == 'primary' and 'editable' in self.trait(item).__dict__ \
                    and not self.trait(item).editable:
                view_block.style = 'readonly'
            if 'position' in self.trait(item).__dict__:
                pos.append(self.trait(item).position)
            else:
                pos.append(None)
            content.append(view_block)
            names.append(item)
        return content, names, pos

    def getViewGroup(self, group_name='status', align='h', setting_group=status_setting,
                            setting_item=status_setting, default_group=None):
        """
        get GUI content for the status parameters
        :param kwargs:
            `align`: 'v' (vertical) or 'h' (horizontal). how individual GUI elements are aligned with each other. for
            status group, default is 'h'
            `setting`: dict, used to set traitsui.api.Item properties. default is status_setting
        :return:
            list
        """
        if align == 'h':
            group_1st = HGroup
            group_2nd = VGroup
        else:
            group_1st = VGroup
            group_2nd = HGroup

        params, names, pos = self.get_view_itemlist(group_name, **setting_item)
        # decide how many sub groups are there
        # in case default groups are provided, there are 2 sub groups
        if default_group:
            d_group = []
            for name in default_param_status:
                if name in names:
                    idx = names.index(name)
                    d_group.append(params.pop(idx))
                    names.pop(idx)
            if params:
                groups = [default_group, params]
            else:
                groups = [default_group]
        # if not, generate groups from pos parameter
        else:
            # first check if the positions are consistent
            dim = [item.__len__() for item in pos if item]
            if set(dim).__len__() > 1:
                raise ValueError('specified positions cannot be uniquely decided: number of dimensions not match')
            # now check how many groups are there, which is the maximum index at the axis of the alignment
            # if dim is 1 or empty, then only one group
            if not dim or dim[0] == 1:
                groups = [params]
            else:
                if align == 'h': group_idx = 0; item_idx = 1
                else: group_idx = 1; item_idx = 0
                user_pos = [item[group_idx] for item in pos if item]
                n_groups = max(user_pos)
                # equally distribute GUI elements into different groups
                n_item_per_group = -(-params.__len__()//n_groups)
                # pick up those items with pre-defined position
                itemidx_withpos = [[] for i in range(n_groups)]
                itemidx_nopos = []
                for idx, p in enumerate(pos):
                    if p: itemidx_withpos[p[group_idx] - 1].append(idx)
                    else: itemidx_nopos.append(idx)
                # number of un-positioned items need to be put in each group
                n_toput = [n_item_per_group - group.__len__() for group in itemidx_withpos]
                # put un-postioned items in each group
                groups = [[] for i in range(n_groups)]
                for idx, item in enumerate(params):
                    if not pos[idx]:
                        for i, n_remains in enumerate(n_toput):
                            if n_remains:
                                groups[i].append(item)
                                n_toput[i] -= 1
                                break
                # now insert items with pre-defined position
                for j, gidx in enumerate(itemidx_withpos):
                    if gidx:
                        item_withpos = [params[i] for i in gidx]
                        item_pos = [pos[i] for i in gidx]
                        ordered_group = [zip_item for zip_item in
                                         sorted(zip(item_pos, item_withpos), key=lambda p: p[0][item_idx])]
                        for item in ordered_group:
                            # need to consider the case where blank space is required
                            while groups[j].__len__() < item[0][item_idx] - 1:
                                groups[j].append(Item('_'))
                            groups[j].insert(item[0][item_idx] - 1, item[1])

        # construct View group on parameter groups
        if groups.__len__() == 1:
            return group_1st(content=groups[0], label=group_name, **setting_group)
        else:
            view_groups = [group_1st(content=item_group) for item_group in groups]
            return group_2nd(*view_groups, label=group_name, **setting_group)

    def copy_values_to(self, destiny=None, names=None):
        """
        HasTraits.copy_traits seems not working. a replacement for the function
        :param destiny: a dict or HasTraits instance to receive the copied values. if not provided create an empty one
        :param names: list of str, name of the attribute to be copied. by default use self.get_parameters()
        :return: destiny
        """
        if not destiny:
            destiny = {}
        if not names:
            names = self.get_parameters()
        for item in names:
            destiny[item] = getattr(self, item)
        return destiny

    def copy_values_from(self, target, names=None):
        """
        HasTraits.copy_traits seems not working. a replacement for the function
        :param target: a dict or HasTraits instance to copy values from
        :param names: list of str, name of the attribute to be copied. by default use self.get_parameters()
        :return: None
        """
        if not names:
            names = self.get_parameters()
        for item in names:
            if isinstance(target, HasTraits):
                setattr(self, item, getattr(target, item))
            elif isinstance(target, dict):
                setattr(self, item, target[item])
            else:
                raise ValueError('target with type of {} is not supported'.format(target.__class__))


class ExperimentSetting(Setting):
    """
    Parameters for ExperimentLogic class
    """

    # every setting class should have those parameters
    category = 'experiment'
    experiment_name      = Str('experiment', group='status', dsec='name of the experiment', noshow=True)
    trial_number         = CInt(0, group='primary', dsec='Number of trials in each condition', reinit=False)
    trial_duration       = CFloat(0, group='primary', dsec='Duration of each trial, (s)', reinit=False)
    inter_trial_interval = CFloat(0, group='primary', dsec='Duration of inter-trial interval, (s)', reinit=False)

    total_duration       = Property(Float, group='status', depends_on=
                ['trial_number', 'trial_duration', 'inter_trial_interval'],
                dsec='Total time of the experiment, (s)')
    start_time           = Instance(datetime.datetime, group='status', dsec='Starting time of the experiment')
    current_trial        = CInt(0, group='status', dsec='Current trial number')
    remain_time          = Instance(datetime.time, group='status', dsec='Remaining time until finish')

    total_trial          = Property(Int, group='status', depends_on=[''],
                                    dsec='Total number of trials')

    operating_mode = Enum('thread', 'normal', group='primary', dsec=
                          'normal mode will block the main thread', noshow=True)

    def _get_total_duration(self):
        return self.trial_number*(self.trial_duration + self.inter_trial_interval)

    def _get_total_trial(self):
        return self.trial_number


class DeviceSetting(Setting):
    """
    Parameters for Device class
    in order to standardize input/output and device setting, all the device setting should have the following primary
    fields:
        type: str, nature of the signal, e.g. 'analog_signal', 'image'
        dtype: data type, preferably a numpy.dtype
        shape: tuple, shape of the data, first dimension is -1 or 0 so it is enlargeable
        sampling_freq: float, sampling frequency in Hz
    """
    # TODO: may be better to have separate input and output settings

    category = 'device'
    device_type = Str(group='status', dsec='type of the device')
    device_ID   = CInt(group='status', dsec='ID of the device')
    device_name = Property(Str, group='status', dsec='name of the device',
                           depends_on=['device_type', 'device_ID'])

    operating_mode = Enum('thread', 'normal', 'subprocess', group='primary', dsec=
                          'normal mode will block the main thread', noshow=True)

    @cached_property
    def _get_device_name(self):
        return self.device_type + '_' + str(self.device_ID)


if __name__ == '__main__':
    t = ExperimentSetting()
    t.configure_traits()
