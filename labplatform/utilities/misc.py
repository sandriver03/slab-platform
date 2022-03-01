'''
    ultilities and helpers

'''

from labplatform.core.Setting import ExperimentSetting

from traits.api import HasTraits, Instance, Button, Any, Int, Str, Enum, Event, Bool, Directory, \
    on_trait_change, Property
from traitsui.api import Handler, UIInfo, View, Item, HGroup, spring, Controller, VGroup, \
    Tabbed, Include, InstanceEditor, Label, OKCancelButtons
from traitsui.message import error
import screeninfo

import logging
log = logging.getLogger(__name__)


def getScreenSize():
    monitors = screeninfo.get_monitors()
    return [[s.width, s.height, s.width_mm, s.height_mm, s.name] for s in monitors]


class ToolBar(HasTraits):

    handler = Instance(Handler)
    info = Instance(UIInfo)

    def install(self, handler, info):
        self.handler = handler
        self.info = info

    def _anytrait_changed(self, trait, value):
        # Discard any trait changes that are not due to the buttons defined in
        # the subclasses.
        if trait not in ('trait_added', 'info', 'handler'):
            getattr(self.handler, trait)(self.info)


class ExperimentToolBar(ToolBar):

    size    = 24, 24
    kw      = dict(height=size[0], width=size[1], action=True)
    apply   = Button('Apply', tooltip='Apply settings', **kw)
    revert  = Button('Revert', tooltip='Revert settings', **kw)
    start   = Button('Run', tooltip='Begin experiment', **kw)
    pause   = Button('Pause', tooltip='Pause', **kw)
    resume  = Button('Resume', tooltip='Resume', **kw)
    stop    = Button('Stop', tooltip='stop', **kw)
    remind  = Button('Remind', tooltip='Remind', **kw)
    cancel_remind = Button('Cancel Remind', tooltip='Remind', **kw)
    item_kw = dict(show_label=False)

    traits_view = View(
            HGroup(Item('apply',
                        enabled_when="object.handler.pending_changes",
                        **item_kw),
                   Item('revert',
                        enabled_when="object.handler.pending_changes",
                        **item_kw),
                   Item('start',
                        enabled_when="object.handler.state=='halted'",
                        **item_kw),
                   '_',
                   Item('remind',
                        enabled_when="object.handler.state=='paused'",
                        **item_kw),
                   Item('cancel_remind',
                        enabled_when="object.handler.state=='paused'",
                        **item_kw),
                   Item('pause',
                        enabled_when="object.handler.state=='running'",
                        **item_kw),
                   Item('resume',
                        enabled_when="object.handler.state=='paused'",
                        **item_kw),
                   Item('stop',
                        enabled_when="object.handler.state in " +\
                                     "['running', 'paused', 'manual']",
                        **item_kw),
                   spring,
                   springy=True,
                   ),
            kind='subpanel',
            )


# blank hastraits object
class TraitObj(HasTraits):
    pass


# -----------------test example----------------------------
# model
class ToolBarTest(HasTraits):
    data0 = Str('', editable=True, group='primary')
    data1 = Str('', editable=True, group='derived')
    data2 = Str('', editable=False, group='primary')
    data3 = Str('', editable=False, group='derived')

    kw = {'show_border':True}

    def get_editable_group(self):
        tl = []

    def get_nonedit_group(self):
        tl = []

    Vgroup_editable = VGroup([Item('data0'), Item('data1')], **kw)
    Vgroup_none     = VGroup([Item('data2'), Item('data3')], **kw)

    traits_view = View()
    traits_view.set_content([Vgroup_editable, Vgroup_none])


# controller
class myController(Controller):

    toolbar = Instance(ExperimentToolBar())
    state   = Enum('halted', 'paused', 'running', 'stopped', 'manual')

    # display state in status when it is changed
    def _state_changed(self):
        self.model.status = self.state

    def init(self, info):
        self.toolbar = ExperimentToolBar()
        self.toolbar.install(self, info)
        self.model = info.object
        self.model.status = self.state

    def start(self, info):
        self.state = 'running'
        print('starting...now state is {}'.format(self.state))

    def stop(self, info):
        self.state = 'stopped'
        print('stopping...now state is {}'.format(self.state))

    def pause(self, info):
        self.state = 'paused'
        print('pausing...now state is {}'.format(self.state))

    def resume(self, info):
        self.state = 'running'
        print('resuming...now state is {}'.format(self.state))

    def close(self, info, is_ok):
        '''
        Prevent user from closing window while an experiment is running since
        data is not saved to file until the stop button is pressed.
        '''
        # We can abort a close event by returning False.  If an experiment
        # is currently running, confirm that the user really did want to close
        # the window.  If no experiment is running, then it's OK since the user
        # can always restart the experiment.
        close = True
        if self.state not in ('stopped', 'halted'):
            mesg = 'Experiment is still running.  Are you sure you want to exit?'
            # The function confirm returns an integer that represents the
            # response that the user requested.  YES is a constant (also
            # imported from the same module as confirm) corresponding to the
            # return value of confirm when the user presses the "yes" button on
            # the dialog.  If any other button (e.g. "no", "abort", etc.) is
            # pressed, the return value will be something other than YES and we
            # will assume that the user has requested not to quit the
            # experiment.
            if error(mesg) != True:
                close = False
            else:
                self.stop(info)

        if close:
            #Handler.close(self, info, is_ok)
            #if self.physiology_handler is not None:
                #print 'attemting to close handler'
                #self.physiology_handler.close(info, True, True)
            pass
        return close

    # trait change handler
    @on_trait_change('model.setting.+context')
    def handle_parameter_change(self, name, old, new):
        #print(self.model.setting.d_para)
        #print(self.info.ui.__dict__)
        print('Attribute {} has changed from {} to {}'.format(name, old, new))


# experiment parameters
class exp_setting(ExperimentSetting):
    para0 = Str('', editable=True, group='primary', dsec='parameter 0')
    para1 = Str('', editable=True, group='primary', dsec='parameter 1')
    para2 = Str('', editable=True, group='derived', dsec='parameter 2')
    para3 = Str('', editable=True, group='derived', dsec='parameter 3')

    d_para = Property(Str, group='derived', dsec='test', depends_on=['para0', 'para1'])

    def _get_d_para(self):
        return self.para1 + self.para0


# view
class experiment(HasTraits):
    setting = Instance(exp_setting, ())
    data    = Instance(ToolBarTest, ())
    status  = Str()

    # define views
    traits_group = VGroup(
        Item('status', style='readonly', show_label=False),
        Tabbed(
            Item('setting', style='custom', show_label=False),
            Item('data', style='custom', show_label=False),
            #Item('handler.tracker', style='custom'),
        ),
        Item('handler.toolbar', style='custom', show_label=False),
    )

    traits_view = View(
        Include('traits_group'),
        resizable=True
    )


if __name__ == '__main__':
    e = experiment()
    ec = myController()
    e.configure_traits(handler=ec)
