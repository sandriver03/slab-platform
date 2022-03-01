from .streamhelpers import DataSender, DataReceiver, register_transfermode
from labplatform.utilities.RingBuffer import RingBuffer

import struct
import numpy as np


class SharedMemSender(DataSender):
    """Stream sender that uses shared memory for efficient interprocess
    communication. Only the data pointer is sent over the socket.

    Note: this class is usually not instantiated directly; use
    ``OutputStream.configure(transfermode='sharedmem')``.

    Extra parameters accepted when configuring the output stream:

    * buffer_size (int) the size of the shared memory buffer in *frames*.
      The total shape of the allocated buffer is ``(buffer_size,) + shape``.
    * double (bool) if True, then the buffer size is doubled and all frames are
      written to the buffer twice. This makes it possible to guarantee
      zero-copy reads by any connected InputStream.
    * axisorder (tuple) The order that buffer axes should be arranged in
      memory. This makes it possible to optimize for specific algorithms that
      expect either row-major or column-major alignment. The default is
      row-major; the time axis comes first in the axis order.
    * fill (float) Value used to fill the buffer where no data is available.
    """

    def __init__(self, socket, params):
        DataSender.__init__(self, socket, params)
        self.size = self.params['buffer_size']
        shape = (self.size,) + tuple(self.params['shape'][1:])
        self.buffer = RingBuffer(shape=shape, dtype=self.params['dtype'],
                                  shmem=True, axisOrder=self.params['axisorder'],
                                  double=self.params['double'], fill=self.params['fill'])
        self.params['shm_id'] = self.buffer.shm_id

    def send(self, index, data, header, **kwargs):
        assert data.dtype == self.params['dtype']
        shape = data.shape
        if self.params['shape'][0] != -1:
            assert shape == self.params['shape']
        else:
            assert tuple(shape[1:]) == tuple(self.params['shape'][1:]), '{} {}'.format(shape, self.params['shape'])

        self.buffer.write(data, index)

        stat = struct.pack('!' + 'QQ' + self.header_string, index, shape[0], *header)
        self.socket.send_multipart([stat])


class SharedMemReceiver(DataReceiver):
    def __init__(self, socket, params):
        # init data receiver with no ring buffer; we will implement our own from shm.
        DataReceiver.__init__(self, socket, params)

        self.size = self.params['buffer_size']
        shape = (self.size,) + tuple(self.params['shape'][1:])
        self.buffer = RingBuffer(shape=shape, dtype=self.params['dtype'], double=self.params['double'],
                                 shmem=self.params['shm_id'], axisOrder=self.params['axisorder'])

    def recv(self, return_data=False):
        """Receive message indicating the index of the next data chunk.

        Parameters:
        -----------
        return_data : bool
            If True, return the new data chunk (this may involve copying data
            from the shared ring buffer). If False, then return None in place
            of data (the new data can still be accessed using __getitem__). The
            default is False.
        """
        stat = self.socket.recv_multipart()[0]
        s_data = struct.unpack('!QQ' + self.header_string, stat)
        index = s_data[0]
        size = s_data[1]
        header = s_data[2:]
        if return_data:
            data = self.buffer[index - size:index]
        else:
            data = None
        return index, data, header


register_transfermode('sharedmem', SharedMemSender, SharedMemReceiver)
