from labplatform.utilities.RingBuffer import RingBuffer
from .streamhelpers import all_transfermodes, MonitorSender, MonitorReceiver
# from ..rpc import ObjectProxy
from labplatform.utilities.arraytools import make_dtype

import random
import string
import zmq
import numpy as np
import weakref


default_stream = dict(
    protocol='tcp',
    interface='127.0.0.1',
    port='*',
    transfermode='plaindata',
    type='analogsignal',
    dtype='float',
    shape=(-1, 1),
    axisorder=None,
    buffer_size=0,
    compression='',
    scale=None,
    offset=None,
    units='',
    sampling_freq=1.,
    double=False,  # make sense only for transfermode='sharemem',
    fill=None,
    header_format='Qs',   # additional information send together with data as header;
                          # should be a format string (see struct.pack)
    header_default=(b'0', ),   # default header value if not provided; must be an instance of list, tuple or
                               # np.ndarray
    header_encoding='utf-8',  # default encoding for strings in the header
    monitor=False,   # if include an additional receiver to monitor the latest state of the output
    monitor_port='*',
)


class OutputStream(object):
    """Class for streaming data to an InputStream.

    Streams allow data to be sent between objects that may exist on different
    threads, processes, or machines. They offer a variety of transfer methods
    including TCP for remote connections and IPC for local connections.
    """

    def __init__(self, spec=None, node=None, name=None):
        spec = {} if spec is None else spec
        self.last_index = 0
        self.configured = False
        self.spec = spec  # this is a priori stream params, and must be change when Node.configure
        if node is not None:
            self.node = weakref.ref(node)
        else:
            self.node = None
        self.name = name
        self.N_packet = 0   # number of packet sent
        self.monitor_socket = None
        self.monitor_url = None
        self._monitored = False
        self.params = None
        self.header_default = None

    def configure(self, **kargs):
        """
        Configure the output stream.

        Parameters
        ----------
        protocol : 'tcp', 'udp', 'inproc' or 'inpc' (linux only)
            The type of protocol used for the zmq.PUB socket
        interface : str
            The bind adress for the zmq.PUB socket
        port : str
            The port for the zmq.PUB socket
        transfermode: str
            The method used for data transfer:

            * 'plaindata': data are sent over a plain socket in two parts: (frame index, data).
            * 'sharedmem': data are stored in shared memory in a ring buffer and the current frame index is sent over the socket.
            * 'shared_cuda_buffer': (planned) data are stored in shared Cuda buffer and the current frame index is sent over the socket.
            * 'share_opencl_buffer': (planned) data are stored in shared OpenCL buffer and the current frame index is sent over the socket.

            All registered transfer modes can be found in `pyacq.core.stream.all_transfermodes`.
        type: 'analogsignal', 'digitalsignal', 'event' or 'image/video'
            The nature of data to be transferred.
        dtype: str ('float32','float64', [('r', 'uint16'), ('g', 'uint16'), , ('b', 'uint16')], ...)
            The numpy.dtype of the data buffer. It can be a composed dtype for event or images.
        shape: list
            The shape of each data frame. If the stream will send chunks of variable length,
            then use -1 for the first (time) dimension.

            * For ``type=image``, the shape should be ``(-1, H, W)`` or ``(n_frames, H, W)``.
            * For ``type=analogsignal`` the shape should be ``(n_samples, n_channels)`` or ``(-1, n_channels)``.
        compression: '', 'blosclz', 'blosc-lz4'
            The compression for the data stream. The default uses no compression.
        scale: float
            An optional scale factor + offset to apply to the data before it is sent over the stream.
            ``output = offset + scale * input``
        offset:
            See *scale*.
        units: str
            Units of the stream data. Mainly used for 'analogsignal'.
        sampling_freq: float or None
            Sample rate of the stream in Hz.
        kwargs :
            All extra keyword arguments are passed to the DataSender constructor
            for the chosen transfermode (for example, see
            :class:`SharedMemSender <stream.sharedmemstream.SharedMemSender>`).
        """
        # make sure params.header_default is in required format
        if 'header_default' in kargs and not isinstance(kargs['header_default'], (list, tuple, np.ndarray)):
            raise ValueError("parameter 'header_default' must be an instance of (list, tuple, np.ndarray), but an "
                             "instance of {} is provided".format(kargs['header_default'].__class__))

        self.params = dict(default_stream)
        self.params.update(self.spec)
        for k in kargs:
            if k in self.spec:
                assert kargs[k] == self.spec[k], \
                    'Cannot configure {}={}; already in fixed in self.spec {}={}'.format(k, kargs[k], k, self.spec[k])
        self.params.update(kargs)
        self.header_default = self.params['header_default']

        shape = self.params['shape']
        assert shape[0] == -1 or shape[0] > 0, "First element in shape must be -1 or > 0."
        for i in range(1, len(shape)):
            assert shape[i] > 0, "Shape index %d must be > 0." % i

        if 'monitor' in self.params and self.params['monitor']:
            self._monitored = True

        if self.params['protocol'] in ('inproc', 'ipc'):
            pipename = u'pyacq_pipe_' + ''.join(
                random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(24))
            self.params['interface'] = pipename
            self.url = '{protocol}://{interface}'.format(**self.params)
            if self._monitored:
                pipename = u'pyacq_pipe_' + ''.join(
                    random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(24))
                self.params['monitor_interface'] = pipename
                self.monitor_url = '{protocol}://{monitor_interface}'.format(**self.params)
        else:
            self.url = '{protocol}://{interface}:{port}'.format(**self.params)
            if self._monitored:
                self.monitor_url = '{protocol}://{interface}:{monitor_port}'.format(**self.params)
        context = zmq.Context.instance()
        self.socket = context.socket(zmq.PUB)
        self.socket.linger = 1000  # don't let socket deadlock when exiting
        self.socket.bind(self.url)
        self.addr = self.socket.getsockopt(zmq.LAST_ENDPOINT).decode()
        self.port = self.addr.rpartition(':')[2]
        self.params['port'] = self.port
        if self._monitored:
            self.monitor_socket = context.socket(zmq.PUB)
            self.monitor_socket.linger = 1000  # don't let socket deadlock when exiting
            self.monitor_socket.bind(self.monitor_url)
            self.monitor_addr = self.monitor_socket.getsockopt(zmq.LAST_ENDPOINT).decode()
            self.monitor_port = self.monitor_addr.rpartition(':')[2]
            self.params['monitor_port'] = self.monitor_port

        transfermode = self.params['transfermode']
        if transfermode not in all_transfermodes:
            raise ValueError("Unsupported transfer mode '%s'" % transfermode)
        sender_class = all_transfermodes[transfermode][0]
        self.sender = sender_class(self.socket, self.params)
        if self._monitored:
            self.monitor_sender = MonitorSender(self.monitor_socket, self.params)

        self.configured = True
        if self.node and self.node():
            self.node()._after_output_configure(self.name)

    def send(self, data, index=None, header=None, **kwargs):
        """Send a data chunk and its frame index.

        Parameters
        ----------
        index: int
            The absolute sample index. This is the index of the last sample + 1.
        data: np.ndarray or bytes
            The chunk of data to send.
        header: tuple or list
            Additional metadata sent together with data. Data type should match what is specified in params.header
            format string
        """
        if index is None:
            index = self.last_index + data.shape[0]
        if header:
            header = (self.N_packet + 1, ) + header
        else:
            header = (self.N_packet + 1, ) + self.header_default
        self.last_index = index
        # type check for np array
        if isinstance(data, np.ndarray) and data.dtype != self.params['dtype']:
            data = data.astype(self.params['dtype'])
        self.sender.send(index, data, header, **kwargs)
        if self._monitored:
            self.monitor_sender.send(header, **kwargs)
        self.N_packet += 1

    def send_monitor_info(self, header=None, **kwargs):
        """Send header information used to monitor the state of the stream

        Parameters
        ----------
        header: tuple or list
            Additional metadata sent together with data. Data type should match what is specified in params.header
            format string
        """
        if header is None:
            header = (self.N_packet + 1,) + self.header_default
        self.monitor_sender.send(header, **kwargs)

    def close(self):
        """Close the output.

        This closes the socket and releases shared memory, if necessary.
        """
        self.sender.close()
        self.socket.close()
        del self.socket
        del self.sender


def _shape_equal(shape1, shape2):
    """
    Check if shape of stream are compatible.
    More or less shape1==shape2 but deal with:
      * shape can be list or tuple
      * shape can have one dim with -1
    """
    shape1 = list(shape1)
    shape2 = list(shape2)
    if len(shape1) != len(shape2):
        return False

    for i in range(len(shape1)):
        if shape1[i] == -1 or shape2[i] == -1:
            continue
        if shape1[i] != shape2[i]:
            return False

    return True


class InputStream(object):
    """Class for streaming data from an OutputStream.

    Streams allow data to be sent between objects that may exist on different
    threads, processes, or machines. They offer a variety of transfer methods
    including TCP for remote connections and IPC for local connections.

    Typical InputStream usage:

    1. Use :func:`InputStream.connect()` to connect to an :class:`OutputStream`
       defined elsewhere. Usually, the argument will actually be a proxy to a
       remote :class:`OutputStream`.
    2. Poll for incoming data packets with :func:`InputStream.poll()`.
    3. Receive the next packet with :func:`InputStream.recv()`.

    Optionally, use :func:`InputStream.set_buffer()` to attach a
    :class:`RingBuffer` for easier data handling.
    """

    def __init__(self, spec=None, node=None, name=None):
        self.spec = {} if spec is None else spec
        self.configured = False
        if node is not None:
            self.node = weakref.ref(node)
        else:
            self.node = None
        self.name = name
        self.buffer = None
        self._own_buffer = False  # whether InputStream should populate buffer
        self.connected = False
        self.N_packet = 0   # number of received packets
        self.latest_packet_N = None  # index of last packet queued in the socket
        self.Idx_initial_packet = 0  # index of the first packet received by the stream
        self.monitor_socket = None
        self.monitor_receiver = None
        self.socket = None
        self.receiver = None

    def connect(self, output, topic=(b'', ), monitor=False, N_packet=0):
        """Connect an output to this input.

        Any data send over the stream using :func:`output.send() <OutputStream.send>`
        can be retrieved using :func:`input.recv() <InputStream.recv>`.

        Parameters
        ----------
        output : OutputStream (or proxy to a remote OutputStream)
            The OutputStream to connect.
        topic: list or tuple, the topic for the socket to subscribe
        monitor: Boolean, if create an additional socket to monitor the input
        N_packet: int, number of packet already sent when connected; only useful with monitored stream
        """
        if isinstance(output, dict):
            self.params = output
            if 'N_packet' not in self.params.keys():
                self.params['N_packet'] = N_packet
        elif isinstance(output, OutputStream):
            self.params = output.params
            self.params['N_packet'] = output.N_packet
        # elif isinstance(output, ObjectProxy):
           # self.params = output.params._get_value()
        else:
            raise TypeError("Invalid type for stream: %s" % type(output))

        if self.params['protocol'] in ('inproc', 'ipc'):
            self.url = '{protocol}://{interface}'.format(**self.params)
            if 'monitor' in self.params and self.params['monitor']:
                self.monitor_url = '{protocol}://{monitor_interface}'.format(**self.params)
        else:
            self.url = '{protocol}://{interface}:{port}'.format(**self.params)
            if 'monitor' in self.params and self.params['monitor']:
                self.monitor_url = '{protocol}://{interface}:{monitor_port}'.format(**self.params)

        # allow some keys in self.spec to override self.params
        readonly_params = ['protocol', 'transfermode', 'shape', 'dtype']
        for k, v in self.spec.items():
            if k in readonly_params:
                if k == 'shape':
                    valid = _shape_equal(v, self.params[k])
                elif k == 'dtype':
                    # ~ valid = v == self.params[k]
                    valid = make_dtype(v) == make_dtype(self.params[k])
                else:
                    valid = (v == self.params[k])
                if not valid:
                    raise ValueError("InputStream parameter %s=%s does not match connected output %s=%s." %
                                     (k, v, k, self.params[k]))
            else:
                self.params[k] = v

        context = zmq.Context.instance()
        self.socket = context.socket(zmq.SUB)
        self.socket.linger = 1000  # don't let socket deadlock when exiting
        for t in topic:
            self.socket.setsockopt(zmq.SUBSCRIBE, t)
        # ~ self.socket.setsockopt(zmq.DELAY_ATTACH_ON_CONNECT,1)
        self.socket.connect(self.url)

        transfermode = self.params['transfermode']
        if transfermode not in all_transfermodes:
            raise ValueError("Unsupported transfer mode '%s'" % transfermode)
        receiver_class = all_transfermodes[transfermode][1]
        self.receiver = receiver_class(self.socket, self.params)

        # add a monitor socket if necessary
        if 'monitor' not in self.params.keys():
            self.params.update({'monitor': monitor})
        if self.params['monitor']:
            self.monitor_socket = context.socket(zmq.SUB)
            self.monitor_socket.setsockopt(zmq.CONFLATE, 1)
            self.monitor_socket.linger = 1000
            for t in topic:
                self.monitor_socket.setsockopt(zmq.SUBSCRIBE, t)
            self.monitor_socket.connect(self.monitor_url)
            self.monitor_receiver = MonitorReceiver(self.monitor_socket, self.params)

        # set N_packet to the output packet number when connected, if available
        if 'N_packet' in self.params.keys():
            self.N_packet = self.params['N_packet']
            self.Idx_initial_packet = self.N_packet

        self.connected = True
        if self.node and self.node():
            self.node()._after_input_connect(self.name)

        if 'offset' in self.params and self.params['offset']:
            self.buffer_offset = self.params['offset']
        else:
            self.buffer_offset = 0

    def poll(self, timeout=0):
        """Poll the socket of input stream. By default returns immediately. Set timeout to None if want blocking

        Return True if a new packet is available.
        """
        return self.socket.poll(timeout=timeout)

    def recv(self, **kargs):
        """
        Receive a chunk of data.

        Returns
        -------
        index: int
            The absolute sample index. This is the index of the last sample + 1.
        data: np.ndarray or bytes
            The received chunk of data.
            If the stream uses ``transfermode='sharedarray'``, then the data is
            returned as None and you must use ``input_stream[start:stop]``
            to read from the shared array or ``input_stream.recv(with_data=True)``
            to return the received data chunk.
        """
        index, data, header = self.receiver.recv(**kargs)
        if self._own_buffer and data is not None and self.buffer is not None:
            self.buffer.write(data, index=index + self.buffer_offset)
        self.N_packet += 1
        return index, data, header

    def monitor(self, **kwargs):
        """
        check the latest sender state with the monitor socket

        Returns:
            index,
            stat
            header
        """
        if self.monitor_socket and self.monitor_socket.poll(0):
            header = self.monitor_receiver.recv(**kwargs)
            self.latest_packet_N = header[0]
            return header

    def close(self):
        """Close the stream.

        This closes the socket. No data can be received after this point.
        """
        self.receiver.close()
        self.socket.close()
        if self.monitor_socket:
            self.monitor_receiver.close()
            self.monitor_socket.close()
            self.monitor_socket = None
        self.connected = False
        # del self.socket

    def __getitem__(self, *args):
        """Return a data slice from the RingBuffer attached to this InputStream.

        If no RingBuffer is attached, raise an exception. See ``set_buffer()``.
        """
        if self.buffer is None:
            raise TypeError("No ring buffer configured for this InputStream.")
        return self.buffer.__getitem__(*args)

    def get_data(self, *args, **kargs):
        """
        Return a segment of the RingBuffer attached to this InputStream.

        If no RingBuffer is attached, raise an exception.

        For parameters, see :func:`RingBuffer.get_data()`.

        See also: :func:`InputStream.set_buffer()`.
        """
        if self.buffer is None:
            if self.receiver.buffer is None:
                raise TypeError("No ring buffer configured for this InputStream.")
            return self.receiver.buffer.get_data(*args, **kargs)
        return self.buffer.get_data(*args, **kargs)

    def set_buffer(self, size=None, double=True, axisorder=None, shmem=None, fill=None):
        """Ensure that this InputStream has a RingBuffer at least as large as
        *size* and with the specified double-mode and axis order.

        If necessary, this will attach a new RingBuffer to the stream and remove
        any existing buffer.
        """
        # first see if we already have a buffer that meets requirements
        bufs = []
        if self.buffer is not None:
            bufs.append((self.buffer, self._own_buffer))
        if self.receiver.buffer is not None:
            bufs.append((self.receiver.buffer, False))
        if not self.buffer or self.buffer.shape[1:] == self.params['shape'][1:]:
            for buf, own in bufs:
                if buf.shape[0] >= size and buf.double == double \
                        and (axisorder is None or all(buf.axisOrder == axisorder)):
                    self.buffer = buf
                    self._own_buffer = own
                    return

        # attach a new buffer
        shape = (size,) + tuple(self.params['shape'][1:])
        dtype = make_dtype(self.params['dtype'])
        self.buffer = RingBuffer(shape=shape, dtype=dtype, double=double, axisOrder=axisorder, shmem=shmem, fill=fill)
        self._own_buffer = True
