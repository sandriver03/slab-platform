'''
    toolbar used in experiment and device controller (GUI environment)

'''

from traits.api import HasTraits, Instance, Button
from traitsui.api import Handler, UIInfo, View, Item, HGroup, spring, Label
import logging
log = logging.getLogger(__name__)


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

    size = 24, 24
    kw = dict(height=size[0], width=size[1], action=True)
    apply = Button('Apply', tooltip='Apply settings', **kw)
    revert = Button('Revert', tooltip='Revert settings', **kw)
    reset = Button('Reset', tooltip='Reset all settings to default', **kw)
    start = Button('Start', tooltip='Begin experiment', **kw)
    pause = Button('Pause', tooltip='Pause', **kw)
    # resume  = Button('Resume', tooltip='Resume', **kw)
    stop = Button('Stop', tooltip='stop', **kw)
    # remind  = Button('Remind', tooltip='Remind', **kw)
    # cancel_remind = Button('Cancel Remind', tooltip='Remind', **kw)
    item_kw = dict(show_label=False)

    traits_view = View(
        HGroup(Item('apply',
                    enabled_when="object.handler._pending_changes",
                    **item_kw),
               Item('revert',
                    enabled_when="object.handler._pending_changes",
                    **item_kw),
               Item('reset',
                    enabled_when="object.handler.state not in ['Running', 'Error']",
                    **item_kw),
               Label('|' * 12),
               Item('start',
                    enabled_when="object.handler.state in ['Ready', 'Paused'] "
                                 "and not object.handler._pending_changes",
                    **item_kw),
               Item('pause',
                    enabled_when="object.handler.state=='Running'",
                    **item_kw),
               Item('stop',
                    enabled_when="object.handler.state not in " + \
                                 "['Running', 'Stopped', 'Error']",
                    **item_kw),
               spring,
               springy=True,
               ),
        kind='subpanel',
    )

    '''
                   Item('remind',
                        enabled_when="object.handler.state=='paused'",
                        **item_kw),
                   Item('cancel_remind',
                        enabled_when="object.handler.state=='paused'",
                        **item_kw),
    '''

