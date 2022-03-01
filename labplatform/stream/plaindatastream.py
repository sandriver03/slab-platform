from .streamhelpers import DataSender, DataReceiver, register_transfermode
from labplatform.utilities.arraytools import decompose_array, make_dtype
from .compression import compress, decompress

import struct
import numpy as np


class PlainDataSender(DataSender):
    """Helper class to send data serialized over socket.

    Note: this class is usually not instantiated directly; use
    ``OutputStream.configure(transfermode='plaindata')``.

    To avoid unnecessary copies (and thus optimize transmission speed), data is
    sent exactly as it appears in memory including array strides.

    This class supports compression.
    """

    def send(self, index, data, header, **kwargs):
        # optional pre-processing before send
        if isinstance(data, np.ndarray):
            for f in self.funcs:
                index, data = f(index, data)

        # serialize
        dtype = data.dtype
        shape = data.shape
        buf, offset, strides = decompose_array(data)

        # compress
        comp = self.params['compression']
        buf = compress(buf, comp, data.itemsize)

        # Pack and send
        stat = struct.pack('!' + 'Q' * (3 + len(shape)) + 'q' * len(strides) + self.header_string,
                           len(shape),
                           index,
                           offset,
                           *(shape + strides),
                           *header, )
        copy = self.params.get('copy', False)
        # self.socket.send_multipart([stat, buf], copy=copy)
        # reverse order of stat and buf so can use stat as mintor
        self.socket.send_multipart([buf, stat], copy=copy)


class PlainDataReceiver(DataReceiver):
    """Helper class to receive data serialized over socket.

    See PlainDataSender.
    """

    def __init__(self, socket, params):
        DataReceiver.__init__(self, socket, params)

    def recv(self, return_data=True):
        # receive and unpack structure
        data, stat = self.socket.recv_multipart()
        ndim = struct.unpack('!Q', stat[:8])[0]
        stat = struct.unpack('!' + 'Q' * (ndim + 2) + 'q' * ndim + self.header_string, stat[8:])
        index = stat[0]

        if not return_data:
            return index, None

        offset = stat[1]
        shape = stat[2:2 + ndim]
        strides = stat[(2 + ndim): (2 + 2 * ndim)]
        header = stat[(2 + 2*ndim):]

        # uncompress
        comp = self.params['compression']
        data = decompress(data, comp)

        # convert to array
        # this avoid some bugs but is not efficient because this is call every sends...
        # dtype = make_dtype(self.params['dtype'])
        dtype = self.params['dtype']
        data = np.ndarray(buffer=data, shape=shape,
                          strides=strides, offset=offset, dtype=dtype)
        return index, data, header


register_transfermode('plaindata', PlainDataSender, PlainDataReceiver)
