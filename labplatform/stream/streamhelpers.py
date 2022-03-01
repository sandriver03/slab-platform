import struct


all_transfermodes = {}


def register_transfermode(modename, sender, receiver):
    global all_transfermodes
    all_transfermodes[modename] = (sender, receiver)


class DataSender:
    """Base class for OutputStream data senders.

    Subclasses are used to implement different methods of data transmission.
    """

    def __init__(self, socket, params):
        self.socket = socket
        self.params = params
        # ~ if isinstance(self.params, ObjectProxy):
        # ~ self.params = self.params._get_value()
        # ~ if 'dtype' in self.params:
        # ~ self.params['dtype'] = make_dtype(self.params['dtype'])
        self.funcs = []
        # header to be sent in addition to the data
        self.header_string = self.params['header_format']
        self.buffer = None

    def send(self, index, data, header, **kwargs):
        raise NotImplementedError()

    def close(self):
        pass


class DataReceiver:
    """Base class for InputStream data receivers.

    Subclasses are used to implement different methods of data transmission.
    """

    def __init__(self, socket, params):
        self.socket = socket
        self.params = params
        # ~ if isinstance(self.params, ObjectProxy):
        # ~ self.params = self.params._get_value()
        # ~ if 'dtype' in self.params:
        # ~ self.params['dtype'] = make_dtype(self.params['dtype'])
        self.buffer = None
        self.header_string = self.params['header_format']

    def recv(self, return_data=False):
        raise NotImplementedError()

    def close(self):
        pass


class MonitorSender(DataSender):

    def send(self, header, **kwargs):

        # Pack and send
        stat = struct.pack('!' + self.header_string,
                           *header, )
        copy = self.params.get('copy', False)
        # self.socket.send_multipart([stat, buf], copy=copy)
        # reverse order of stat and buf so can use stat as mintor
        self.socket.send(stat, copy=copy)


class MonitorReceiver(DataReceiver):

    def recv(self, return_data=False):
        # we should only receive stat from the socket
        stat = self.socket.recv()
        try:
            header = struct.unpack('!' + self.header_string, stat)
        except MemoryError:
            raise

        return header
