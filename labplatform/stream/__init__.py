# -*- coding: utf-8 -*-
# Copyright (c) 2016, French National Center for Scientific Research (CNRS)
# Distributed under the (new) BSD License. See LICENSE for more info.

from .stream import InputStream, OutputStream, default_stream
from labplatform.utilities.RingBuffer import RingBuffer
from labplatform.utilities.sharedArray import SharedArray
from .streamhelpers import all_transfermodes, register_transfermode
from .compression import compression_methods

# import transfer modes so they register their helper classes
from . import plaindatastream
from . import sharedmemstream
