import labplatform.stream as Stream
from labplatform.GUI import Viewer, Viewer_subprocess
import numpy as np
import multiprocessing as mp
import time

"""
    same process
"""
# data viewer with animation
dv_ani = Viewer.DataViewer()
# output stream
outStream = Stream.OutputStream()
# output stream specs; for in-process, use inproc protocol
outStream.configure(protocol='inproc', sampling_freq=1000)
# test input stream
dv_ani.connect_datastream(outStream)


img_dv = Viewer.ImageViewer()
outStream = Stream.OutputStream()
outStream.configure(protocol='inproc', sampling_freq=30, shape=(-1, 1024, 1024))
img_dv.connect_datastream(outStream)

"""
    different process
"""
# data viewer with animation
dv_ani = Viewer.DataViewer()
# output stream
outStream = Stream.OutputStream()
# output stream specs; for in-process, use inproc protocol
outStream.configure(protocol='tcp', sampling_freq=1000)
# test input stream
dv_ani.connect_datastream(outStream)
# run viewer in sub process
dv_ani.configure(operating_mode='subprocess')


img_dv = Viewer.ImageViewer()
outStream = Stream.OutputStream()
outStream.configure(protocol='tcp', sampling_freq=30, shape=(-1, 1024, 1024))
img_dv.connect_datastream(outStream)
img_dv.configure(operating_mode='subprocess')


img_dv = Viewer_subprocess.ImageViewer()
img_dv.configure()
outStream = Stream.OutputStream()
outStream.configure(protocol='tcp', sampling_freq=30, shape=(-1, 1024, 1024))
img_dv.connect_datastream(outStream)


data_dv = Viewer_subprocess.DataViewer()
data_dv.configure()
outStream = Stream.OutputStream()
outStream.configure(protocol='tcp', sampling_freq=1000, shape=(-1, 1))
data_dv.connect_datastream(outStream)