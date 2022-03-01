from traits.api import HasTraits, Instance, Str, Int, List, Any, DelegatesTo
from traitsui.api import Handler, View, Item, Action

"""
 for whatever the reason, after info.ui.dispose() the control is not returned to the command line
"""


class MyPanelHandler(Handler):

    def _process_values(self, info):
        if not info.initialized:
            return # do this in case info takes long to initialize
        # invoke methods
        info.object.model.process_values(info.object.name, info.object.age)
        print('values have been processed')
        info.ui.dispose() # THIS IS WHAT I WAS ACTUALLY ASKING FOR

    def _save_values(self, info):
        if not info.initialized: return
        info.object.model.save_values(info.object.name, info.object.age)
        print('values have been saved')
        info.ui.dispose()


class MyPanel(HasTraits):
    model = Any
    name = Str
    age = Int
    process_values_button = Action(name = 'Process Values', action = '_process_values')
    save_values_button = Action(name = 'Save Params', action = '_save_values')
    view = View( 'name', 'age', handler = MyPanelHandler(),
            buttons = [process_values_button, save_values_button],)


class MyApp(HasTraits):
    panel = Instance(MyPanel)

    def __init__(self):
        self.panel = MyPanel(model = self)

    def get_values(self):
        self.panel.configure_traits()

    def save_values(self, name, age):
        print('... saving (%s, %s)' % (name, age))

    def process_values(self, name, age):
        print('... processing (%s, %s)' % (name, age))


if __name__ == '__main__':
    a = MyApp()
    a.get_values()
