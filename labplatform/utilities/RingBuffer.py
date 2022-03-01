from labplatform.utilities.arraytools import make_dtype
from labplatform.utilities.sharedArray import SharedMem

import numpy as np
import logging

log = logging.getLogger(__name__)


CLS_PARA_LOOKUP = {'shape': 'shape', 'dtype': 'dtype', 'double': 'double',
                   'fill': '_filler', 'axisOrder': 'axis_order'}


class RingBuffer:
    """
    Ring buffer based on numpy array
    the buffer can take any shape, but only writes/reads along the first dimension
    """

    def __init__(self, shape, dtype, double=True, shmem=None, fill=None, axisOrder=None):
        """

        Args:
            shape: tuple, shape of the buffer
            dtype: str or np.dtype
            double: boolean, if use double buffer or not
            shmem: if use share memory as buffer; True create a new
            fill: fill value
            axisOrder: list or tuple, order of axes as written in memory
        """
        self.double = double
        self.shape = shape
        # order of axes as written in memory. This does not affect the shape of the
        # buffer as seen by the user, but can be used to make sure a specific axis
        # is contiguous in memory.
        if axisOrder is None:
            axisOrder = np.arange(len(shape))
        self.axis_order = axisOrder

        shape = (shape[0] * (2 if double else 1), ) + shape[1:]
        native_shape = np.array(shape)[self.axis_order]

        # initialize int buffers with 0 and float buffers with nan
        if fill is None:
            fill = 0 if make_dtype(dtype).kind in 'ui' else np.nan
        self._filler = fill

        # create buffer
        if shmem is None:
            self.buffer = np.empty(native_shape, dtype=make_dtype(dtype)).transpose(np.argsort(axisOrder))
            self.buffer[:] = self._filler
            self._indexes = np.zeros((3,), dtype='int64')
            self._shmem = None
            self.shm_id = None
        else:
            size = int(np.product(shape) * make_dtype(dtype).itemsize + 16)
            if shmem is True:
                # create a new shared memory
                self._shmem = SharedMem(nbytes=size)
            else:
                #
                self._shmem = SharedMem(nbytes=size, shm_id=shmem)
            buf = self._shmem.to_numpy(offset=16, dtype=dtype, shape=native_shape)
            self.buffer = buf.transpose(np.argsort(axisOrder))
            self._indexes = self._shmem.to_numpy(offset=0, dtype='int64', shape=(3, ))
            self.shm_id = self._shmem.shm_id

        self.dtype = self.buffer.dtype
        # set read, write and written index
        if shmem in (None, True):
            self._set_write_index(0)
            self._set_written_index(0)
            self._set_read_index(0)

    def reset(self, **kwargs):
        """
        reset the buffer

        Args:
            **kwargs:
            parameters used in class constructor; use them when you want to change the buffer

        Returns:
            None
        """
        for key in kwargs:
            if key in ('shape', 'dtype', 'fill', 'double', 'axisOrder'):
                self.__setattr__(CLS_PARA_LOOKUP[key], kwargs[key])
        self.__init__(self.shape, self.dtype, self.double, self._filler, self.axis_order)

    def reset_index(self):
        """
        reset all indexes to 0
        Returns:
            None
        """
        if self._shmem:
            self._indexes = self._shmem.to_numpy(offset=0, dtype='int64', shape=(3,))
        else:
            self._indexes = np.zeros((3,), dtype='int64')

    def index(self):
        return self._written_index

    def first_index(self):
        idx = self._written_index - self.shape[0]
        return idx if idx >= 0 else 0

    @property
    def _write_index(self):
        return self._indexes[1]

    @property
    def _written_index(self):
        return self._indexes[0]

    @property
    def _read_index(self):
        return self._indexes[2]

    @property
    def _index_range(self):
        return self.first_index(), self._written_index

    def _set_write_index(self, i):
        # what kind of protection do we need here?
        self._indexes[1] = i

    def _set_written_index(self, i):
        # what kind of protection do we need here?
        self._indexes[0] = i

    def _set_read_index(self, i):
        # what kind of protection do we need here?
        self._indexes[2] = i

    def write(self, data, index=None):
        """
        write a chunk of data into the buffer
        Args:
            data: data to be put
            index: starting position of the buffer to put the data

        Returns:
            None
        """
        if self.shape.__len__() - data.shape.__len__() == 1:
            # when data is image, or anything >= 2d
            if data.shape.__len__() > 1:
                data = data.reshape((1, ) + data.shape)
            # when data is 1d
            elif self.shape[1] == 1:
                data = data.reshape(data.shape + (1, ))
            else:
                raise ValueError('data with shape {} cannot be written in to buffer with shape {}'.
                                 format(data.shape, self.shape))

        dsize = data.shape[0]
        bsize = self.shape[0]
        if dsize > bsize:
            raise ValueError("Data chunk size %d is too large for ring "
                             "buffer of size %d." % (dsize, bsize))
        if data.dtype != self.dtype:
            raise TypeError("Data has incorrect dtype %s (buffer requires %s)" %
                            (data.dtype, self.dtype))

        # by default, index advances by the size of the chunk
        if index is None:
            index = self._write_index + dsize

        assert dsize <= index - self._write_index, ("Data size is %d, but index "
                                                    "only advanced by %d." %
                                                    (dsize, index - self._write_index))

        revert_inds = [self._written_index, self._write_index]
        try:
            # advance write index. This immediately prevents other processes from
            # accessing memory that is about to be overwritten.
            self._set_write_index(index)

            # decide if any skipped data needs to be filled in
            fill_start = max(self._written_index, self._write_index - bsize)
            fill_stop = self._write_index - dsize

            if fill_stop > fill_start:
                # data was skipped; fill in missing regions with 0 or nan.
                self._write_data(fill_start, fill_stop, self._filler)
                revert_inds[1] = fill_stop

            self._write_data(self._write_index - dsize, self._write_index, data)

            self._set_written_index(index)
            # raise warnings when overwriting unread buffer
            if self.first_index() > self._read_index:
                log.debug('(%d) slots in the buffer are overwritten' %
                            (self.first_index() - self._read_index))
                self._set_read_index(self.first_index())
        except:
            # If there is a failure writing data, revert read/write pointers
            self._set_written_index(revert_inds[0])
            self._set_write_index(revert_inds[1])
            raise

    def _write_data(self, start, stop, value):
        # get starting index
        bsize = self.shape[0]
        dsize = stop - start
        i = start % bsize

        if self.double:
            self.buffer[i:i + dsize] = value
            i += bsize

        if i + dsize <= self.buffer.shape[0]:
            self.buffer[i:i + dsize] = value
        else:
            n = self.buffer.shape[0] - i
            self.buffer[i:] = value[:n]
            self.buffer[:dsize - n] = value[n:]

    def __getitem__(self, item):
        """
        return a view of the buffer

        Args:
            item: int or slice

        Returns:
            view of the buffer (np.array)
        """
        if isinstance(item, tuple):
            first = item[0]
            rest = (slice(None),) + item[1:]
        else:
            first = item
            rest = None

        if isinstance(first, (int, np.integer)):
            start = self._interpret_index(first)
            stop = start + 1
            data = self.get_data(start, stop)[0]
            if rest is not None:
                data = data[rest[1:]]
        elif isinstance(first, slice):
            start, stop, step = self._interpret_index(first)
            if start == 0 and start == stop:
                return self.buffer[0:0]
            data = self.get_data(start, stop)[::step]
            if rest is not None:
                data = data[rest]
        else:
            raise TypeError("Invalid index type %s" % type(first))

        return data

    def get_data(self, start, stop, copy=False, join=True):
        """
        Return a segment of the ring buffer.

        Args:
            start: int, the starting index of the segment to return.
            stop: int, the stop index of the segment to return (the sample at this index
                will not be included in the returned data)
            copy: bool, if True, then a copy of the data is returned to ensure that modifying
                the data will not affect the ring buffer. If False, then a reference to
                the buffer will be returned if possible. Default is False.
            join: bool, if True, then a single contiguous array is returned for the entire
                requested segment. If False, then two separate arrays are returned
                for the beginning and end of the requested segment. This can be
                used to avoid an unnecessary copy when the buffer has double=False
                and the caller does not require a contiguous array.

        Returns:
            requested data in the buffer
        """
        first, last = self.first_index(), self.index()
        if start < first or stop > last:
            raise IndexError("Requested segment (%d, %d) is out of bounds for ring buffer. "
                             "Current bounds are (%d, %d)." % (start, stop, first, last))

        bsize = self.shape[0]
        copied = False

        if self.double:
            # This do not work when get_data(-10, 50) meaning stop=50 length=60 (start=stop-length)
            # this is util at the beging to get larger buffer than already possible
            # start_ind = start % bsize
            # stop_ind = start_ind + (stop - start)

            # I prefer this which equivalent but work with start<0:
            stop_ind = stop % bsize + bsize
            start_ind = stop_ind - (stop - start)

            data = self.buffer[start_ind:stop_ind]
        else:
            break_index = self._write_index - (self._write_index % bsize)
            if (start < break_index) == (stop <= break_index):
                start_ind = start % bsize
                stop_ind = start_ind + (stop - start)
                data = self.buffer[start_ind:stop_ind]
            else:
                # need to reconstruct from two pieces
                newshape = np.array((stop - start,) + self.shape[1:])[self.axis_order]
                a = self.buffer[start % bsize:]
                b = self.buffer[:stop % bsize]
                if join is False:
                    if copy is True:
                        return a.copy(), b.copy()
                    else:
                        return a, b
                else:
                    data = np.empty(newshape, self.buffer.dtype).transpose(np.argsort(self.axis_order))
                    data[:break_index - start] = a
                    data[break_index - start:] = b
                    copied = True

        if copy and not copied:
            data = data.copy()

        if join:
            return data
        else:
            empty = np.empty((0,) + data.shape[1:], dtype=data.dtype)
            return data, empty

    def read(self, length=1, start=0, copy=False, join=True):
        """
        short hand method for get_data; by default reading from read_index

        Args:
            start: uint, the starting index of the segment to return, relative to current read_index
            length: int, the length of the segment to return
            copy: bool, if True, then a copy of the data is returned to ensure that modifying
                the data will not affect the ring buffer. If False, then a reference to
                the buffer will be returned if possible. Default is False.
            join: bool, if True, then a single contiguous array is returned for the entire
                requested segment. If False, then two separate arrays are returned
                for the beginning and end of the requested segment. This can be
                used to avoid an unnecessary copy when the buffer has double=False
                and the caller does not require a contiguous array.
        Returns:
            part of the buffer, either as a copy or a view
        """
        c_len = self.index() - self.first_index()
        if length > c_len:
            raise IndexError("Requested segment length (%d) is out of bounds for ring buffer. "
                             "Current buffer length is (%d)." % (length, c_len))

        # read data using `get_data`
        data = self.get_data(self._read_index+start, self._read_index+length, copy, join)
        # update read index
        self._set_read_index(self._read_index + length)
        return data

    def _interpret_index(self, index):
        """
        Return normalized index, accounting for negative and None values.
        Also check that the index is readable.

        Slices are returned such that start,stop are swapped and shifted -1 if
        the step is negative. This makes it possible to collect the result in
        the forward direction and handle the step later.

        Args:
            index: int, index to be interpreted

        Returns:
            interpreted index
        """
        start_index = self.first_index()
        if isinstance(index, (int, np.integer)):
            if index < 0:
                index += self._written_index
            if index >= self._written_index or index < start_index:
                raise IndexError("Index %d is out of bounds for ring buffer [%d:%d]" %
                                 (index, start_index, self._written_index))
            return index
        elif isinstance(index, slice):
            start, stop, step = index.start, index.stop, index.step

            # Handle None and negative steps
            if step is None:
                step = 1
            if step < 0:
                start, stop = stop, start

            # Interpret None and negative indices
            if start is None:
                start = start_index
            else:
                if start < 0:
                    start += self._written_index
                if step < 0:
                    start += 1

            if stop is None:
                stop = self._written_index
            else:
                if stop < 0:
                    stop += self._written_index
                if step < 0:
                    stop += 1

            # Bounds check.
            # Perhaps we could clip the returned data like lists/arrays do,
            # but in this case the feedback is likely to be useful to the user.
            if stop > self._written_index or stop < start_index:
                raise IndexError("Stop index %d is out of bounds for ring buffer [%d, %d]" %
                                 (stop, start_index, self._written_index))
            if start > self._written_index or start < start_index:
                raise IndexError("Start index %d is out of bounds for ring buffer [%d, %d]" %
                                 (start, start_index, self._written_index))
            return start, stop, step
        else:
            raise TypeError("Invalid index %s" % index)