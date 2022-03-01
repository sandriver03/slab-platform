from labplatform.stream.compression import compression_methods
from labplatform.stream.stream import OutputStream, InputStream

import time
import numpy as np
import cProfile
import sys


# data transfer protocols for pyzmq
protocols = ['tcp', 'inproc', 'ipc']  # 'udp' is not working
if sys.platform.startswith('win'):
    protocols.remove('ipc')


def benchmark_stream(protocol, transfermode, compression, chunksize, nb_channels=16, nloop=10, profile=False):
    ring_size = chunksize * 20
    stream_spec = dict(protocol=protocol, interface='127.0.0.1', port='*',
                       transfermode=transfermode, streamtype='analogsignal',
                       dtype='float32', shape=(-1, nb_channels), compression=compression,
                       scale=None, offset=None, units='',
                       # for sharedarray
                       sharedarray_shape=(ring_size, nb_channels), timeaxis=0,
                       ring_buffer_method='double', buffer_size=ring_size,
                       )
    outstream = OutputStream()
    outstream.configure(**stream_spec)
    time.sleep(.5)
    instream = InputStream()
    instream.connect(outstream)

    arr = np.random.rand(chunksize, nb_channels).astype(stream_spec['dtype'])

    perf = []
    prof = cProfile.Profile()
    for i in range(nloop):
        start = time.perf_counter()
        if profile:
            prof.enable()
        outstream.send(arr)
        index2, arr2, header = instream.recv()
        if profile:
            prof.disable()
        perf.append(time.perf_counter() - start)
    if profile:
        prof.print_stats('cumulative')

    outstream.close()
    instream.close()

    dt = np.min(perf)
    print(chunksize, nloop, transfermode, protocol.ljust(6), compression.ljust(13), 'time = %0.02f ms' % (dt * 1000),
          'speed = ', chunksize * nb_channels * 4 * 1e-6 / dt, 'MB/s')

    return dt


if len(sys.argv) > 1 and sys.argv[1] == 'profile':
    benchmark_stream(protocol='inproc', transfermode='plaindata',
                     compression='', chunksize=100000, nb_channels=16,
                     profile=True, nloop=100)


else:
    nb_channels = 32
    for chunksize in [2 ** 10, 2 ** 14, 2 ** 16]:
        print('#' * 5)
        for compression in compression_methods:
            for protocol in protocols:
                benchmark_stream(protocol=protocol, transfermode='plaindata',
                                 compression=compression,
                                 chunksize=chunksize, nb_channels=nb_channels)

        benchmark_stream(protocol='tcp', transfermode='sharedmem', compression='',
                         chunksize=chunksize, nb_channels=nb_channels)
