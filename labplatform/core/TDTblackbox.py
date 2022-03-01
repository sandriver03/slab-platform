# -*- coding: utf-8 -*-
"""
eCreated on Mon Jun 12 12:20:32 2017
@author: ob56dap

TDT ActiveX black box
"""
import win32com.client
import numpy as np
import os
from collections import Counter
from typing import Union
import random
import logging
log = logging.getLogger(__name__)


def initialize_processor(processor=None, connection=None, index=None, path=None):
    # Create object to handle TDT-functions
    try:
        RP = win32com.client.Dispatch('RPco.X')
    except win32com.client.pythoncom.com_error as e:
        print("Error:", e)
        return -1
    print("Successfully initialized TDT ActiveX interface")

    # Get the parameters
    if not processor:
        processor = input("what type of device do you want to connect to")
    if not connection:
        connection = input("is the device connected via 'USB' or 'GB'?")
    if not index:
        index = int(input("whats the index of the device (starting with 1)?"))
    if not path:
        path = input("full path of the circuit you want to load")

    # Connect to the device
    if processor == "RM1":
        if RP.ConnectRM1(connection, index):
            print("Connected to RM1")
    elif processor == "RP2":
        if RP.ConnectRP2(connection, index):
            print("Connected to RP2")
    elif processor == "RX8":
        if RP.ConnectRX8(connection, index):
            print("Connected to RX8")
    elif processor == "RX6":
        if RP.ConnectRX6(connection, index):
            print("Connected to RX6")
    else:
        print("Error: unknown device!")
        return -1

    if not RP.ClearCOF():
        print("ClearCOF failed")
        return -1

    if RP.LoadCOF(path):
        print("Circuit {0} loaded".format(path))
    else:
        print("Failed to load {0}".format(path))
        return -1

    if RP.Run():
        print("Circuit running")
    else:
        print("Failed to run {0}".format(path))
        return -1

    RP.processor = processor
    RP.connection = connection
    RP.index = index
    RP.rcx_path = path
    return RP


def initialize_zbus(connection):

    try:
        ZB = win32com.client.Dispatch('ZBUS.x')
    except win32com.client.pythoncom.com_error as e:
        print("Error:", e)
        return -1
    print("Successfully initialized ZBus")

    if ZB.ConnectZBUS(connection):
        print("Connected to ZBUS")
    else:
        print("failed to connect to ZBUS")

    return ZB


def get_ParTags(RP_handle):
    """
    get ParTags associated with a device controlled by a RPox connection
    Args:
        RP_handle: handle return by initiate_processor

    Returns:
        list of strings
    """
    # get number of ParTags
    n_pt = RP_handle.GetNumOf('ParTag')
    # get names of ParTags
    pt_names = []
    for n in range(n_pt):
        pt_names.append(RP_handle.GetNameOf('ParTag', n + 1))
    return pt_names


def read_buffer(RP_handle, buffer_tag, start_idx, nsamples, dtype):
    """
    read data from a processor buffer

    Args:
        RP_handle: handle to a processor
        buffer_tag: str, name of the buffer tag
        start_idx: first index to be read
        nsamples: length of the buffer to be read, in number of samples
        dtype: data type of the buffer. should be np.dtype

    Returns:
        np array
    """
    # the ReadTagV treats each 32-bit buffer as floats
    data = np.array(RP_handle.ReadTagV(buffer_tag, start_idx, nsamples), dtype=np.float32)
    # use np.ndarray.view to change the representation of dtype
    if np.issubdtype(dtype, np.integer):
        data = data.view(np.int32).astype(dtype)
    return data


def get_dac_delay(processor):
    """
    get digital-analog conversion delay, in number of data samples
    Args:
        processor: string, name of the TDT dac processor

    Returns:
        int
    """
    if processor == 'RP2':
        return 65
    elif processor == 'RX6':
        return 47
    elif processor == 'RX8':
        return 24
    elif processor == 'RZ6':
        return 47
    elif processor is None:
        return 0
    else:
        raise ValueError('processor {} is not known'.format(processor))


def get_adc_delay(processor):
    """
    get analog-digital conversion delay, in number of data samples
    Args:
        processor: string, name of the TDT dac processor

    Returns:
        int
    """
    if processor == 'RP2':
        return 30
    elif processor == 'RX6':
        return 66
    elif processor == 'RX8':
        return 47
    elif processor == 'RZ6':
        return 66
    elif processor is None:
        return 0
    else:
        raise ValueError('processor {} is not known'.format(processor))


def set_variable(var_name, value, proc, offset=0):
    """
    Set a variable on a processor to a value. Setting will silently fail if
    variable does not exist in the rcx file. The function will use
    SetTagVal or WriteTagV correctly, depending on whether
    len(value) == 1 or is > 1

    Args:
        var_name: str, name of the data tag to be set
        value: number or np.array, value to be loaded to the data tag
        proc: device handle of the data tag
        offset: int, buffer offset index

    Returns:

    """
    try:
        proc_name = proc.name
    except AttributeError:
        proc_name = proc.__str__()

    if isinstance(value, (list, np.ndarray)):
        # this is the same as using WriteTagV
        flag = proc._oleobj_.InvokeTypes(
            15, 0x0, 1, (3, 0), ((8, 0), (3, 0), (0x2005, 0)),
            var_name, int(offset), value)
        log.info(f'Set {var_name} on {proc_name} with offset {offset}.')
    else:
        flag = proc.SetTagVal(var_name, value)
        log.info(f'Set {var_name} to {value} on {proc_name}.')

    if flag == 0:
        log.info("Unable to set tag '%s' to value %s on device %s"
               % (var_name, value, proc_name))

    return flag


def trigger(trig=1, proc=None, trig_param=(0, 0, 20)):
    """
    Send a trigger. Options are SoftTrig numbers, "zBusA" or "zBusB".
    For using the software trigger a processor must be specified by name
    or index in _procs. Initialize the zBus befor sending zBus triggers.
    Args:
        trig: int, float or str
        proc: device handle used to send the trigger
        trig_param: 3 element tuple of numbers used for the zBus triggers, (Racknum, Trig type, delay)
                    see ActiveX user manual for details

    Returns:

    """
    if isinstance(trig, (int, float)):
        if not proc:
            raise ValueError('Proc needs to be specified for SoftTrig!')
        proc.SoftTrg(trig)
        try:
            proc_name = proc.name
        except AttributeError:
            proc_name = proc.__str__()
        log.info(f'SoftTrig {trig} sent to {proc_name}.')
    elif 'zbus' in trig.lower():
        if trig.lower() == "zbusa":
            proc.zBusTrigA(*trig_param)
            log.info('zBusA trigger sent.')
        elif trig.lower() == "zbusb":
            proc.zBusTrigB(*trig_param)
            log.info('zBusB trigger sent.')
    else:
        raise ValueError("Unknown trigger type! Must be 'soft', "
                         "'zBusA' or 'zBusB'!")


class Processors(object):
    """
    Class for handling initialization of and basic input/output to TDT-processors.
    Methods include: initializing processors, writing and reading data, sending
    triggers and halting the processors.
    """

    def __init__(self):
        self.procs = dict()
        self.mode = None
        self._zbus = None

    def initialize(self, proc_list, zbus=False, connection='GB'):
        """
        Establish connection to one or several TDT-processors.
        Initialize the processors listed in proc_list, which can be a list
        or list of lists. The list / each sublist contains the name and model
        of a processor as well as the path to an rcx-file with the circuit that is
        run on the processor. Elements must be in order name - model - circuit.
        If zbus is True, initialize the ZBus-interface. If the processors are
        already initialized they are reset
        Args:
            proc_list : each sub-list represents one
                processor. Contains name, model and circuit in that order
            zbus : if True, initialize the Zbus interface.
            connection: type of connection to processor, can be "GB" (optical) or "USB"
        Examples:
        #    >>> devs = Processors()
        #    >>> # initialize a processor of model 'RP2', named 'RP2' and load
        #    >>> # the circuit 'example.rcx'. Also initialize ZBus interface:
        #    >>> devs.initialize_processors(['RP2', 'RP2', 'example.rcx'], True)
        #    >>> # initialize two processors of model 'RX8' named 'RX81' and 'RX82'
        #    >>>devs.initialize_processors(['RX81', 'RX8', 'example.rcx'],
        #    >>>                        ['RX82', 'RX8', 'example.rcx'])
        """
        # TODO: check if names are unique and id rcx files do exist
        logging.info('Initializing TDT processors, this may take a moment ...')
        models = []
        if not all([isinstance(p, list) for p in proc_list]):
            proc_list = [proc_list]  # if a single list was provided, wrap it in another list
        for name, model, circuit in proc_list:
            # advance index if a model appears more then once
            models.append(model)
            index = Counter(models)[model]
            print(f"initializing {name} of type {model} with index {index}")
            self.procs[name] = self._initialize_proc(model, circuit,
                                                     connection, index)
        if zbus:
            self._zbus = self._initialize_zbus(connection)
        if self.mode is None:
            self.mode = "custom"

    def write(self, tag, value, procs, offset=0):
        """
        Write data to processor(s).
        Set a tag on one or multiple processors to a given value. Processors
        are addressed by their name (the key in the _procs dictionary). The same
        tag can be set to the same value on multiple processors by passing a
        list of names.
        This function will call SetTagVal or WriteTagV depending on whether
        value is a single integer or float or an array. If the tag could
        not be set (there are different reasons why that might be the case) a
        warning is triggered. CAUTION: If the data type of the value arg does
        not match the data type of the tag, write might be successful but
        the processor might behave strangely.
        Args:
            tag : name of the tag in the rcx-circuit where value is
                written to
            value : value that is written to the tag. Must
                match the data type of the tag.
            procs : name(s) of the processor(s) to write to
            offset: int, buffer offset when applicable
        Examples:
        #    >>> # set the value of tag 'data' on RX81 & RX82 to 0
        #    >>> write('data', 0, ['RX81', 'RX82'])
        """
        if isinstance(value, (np.int32, np.int64)):
            value = int(value)  # use built-int data type
        if isinstance(procs, str):
            if procs == "RX8s":
                procs = [proc for proc in self.procs.keys() if "RX8" in proc]
            elif procs == "all":
                procs = list(self.procs.keys())
            else:
                procs = [procs]
        # Check if the procs are actually there
        if not set(procs).issubset(self.procs.keys()):
            raise ValueError('Can not find some of the specified processors!')
        flags = []
        for proc in procs:
            flags.append(set_variable(tag, value, proc, offset))
        return flags

    def read(self, tag, proc, n_samples=1, start_idx=0, dtype=np.float):
        """
        Read data from processor.
        Get the value of a tag from a processor. The number of samples to read
        must be specified, default is 1 which means reading a single float or
        integer value. Unlike in the write method, reading multiple variables
        in one call of the function is not supported.
        Args:
            tag: name of the processor to write to
            proc: processor to read from
            n_samples: number of samples to read from processor, default=1
            start_idx: int, starting index to read
            dtype: np.dtype, data type of the buffer to be read
        Returns:
            type (int, float, list): value read from the tag
        """
        if n_samples > 1:
            value = read_buffer(self.procs[proc], tag, start_idx, n_samples, dtype)
        else:
            value = self.procs[proc].GetTagVal(tag)
        logging.info(f'Got {tag} from {proc}.')
        return value

    def halt(self):
        """
        Halt all currently active processors.
        """
        # TODO: can we see if halting was successfull
        for proc_name in self.procs.keys():
            proc = self.procs[proc_name]
            if hasattr(proc, 'Halt'):
                logging.info(f'Halting {proc_name}.')
                proc.Halt()

    def trigger(self, trig=1, proc=None, trig_param=(0, 0, 20)):
        """
        Send a trigger. Options are SoftTrig numbers, "zBusA" or "zBusB".
        For using the software trigger a processor must be specified by name
        or index in _procs. Initialize the zBus befor sending zBus triggers.
        Args:
            trig: int, float or str
            proc: str, name of the device used to send the trigger
            trig_param: 3 element tuple of numbers used for the zBus triggers, (Racknum, Trig type, delay)
                            see ActiveX user manual for details

        Returns:

        """
        if isinstance(proc, str):
            device_handle = self.procs[proc]
        else:
            device_handle = proc
        trigger(trig=trig, proc=device_handle, trig_param=trig_param)

    @staticmethod
    def _initialize_proc(model: str, circuit: str, connection: str, index: int):
        return initialize_processor(processor=model, connection=connection, index=index,
                                    path=circuit)

    @staticmethod
    def _initialize_zbus(connection: str = "GB"):
        return initialize_zbus(connection=connection)


class _COM:
    """
    Working with TDT processors is only possible on windows machines. This dummy class
    simulates the output of a processor to test code on other operating systems
    """
    @staticmethod
    def ConnectRX8(connection: str, index: int) -> int:
        if connection not in ["GB", "USB"]:
            return 0
        if not isinstance(index, int):
            return 0
        else:
            return 1

    @staticmethod
    def ConnectRP2(connection: str, index: int) -> int:
        if connection not in ["GB", "USB"]:
            return 0
        if not isinstance(index, int):
            return 0
        else:
            return 1

    @staticmethod
    def ConnectRM1(connection: str, index: int) -> int:
        if connection not in ["GB", "USB"]:
            return 0
        if not isinstance(index, int):
            return 0
        else:
            return 1

    @staticmethod
    def ConnectRX6(connection: str, index: int) -> int:
        if connection not in ["GB", "USB"]:
            return 0
        if not isinstance(index, int):
            return 0
        else:
            return 1

    @staticmethod
    def ClearCOF() -> int:
        return 1

    @staticmethod
    def LoadCOF(circuit: str) -> int:
        if not os.path.isfile(circuit):
            return 0
        else:
            return 1

    @staticmethod
    def Run() -> int:
        return 1

    @staticmethod
    def ConnectZBUS(connection: str) -> int:
        if connection not in ["GB", "USB"]:
            return 0
        else:
            return 1

    @staticmethod
    def Halt() -> int:
        return 1

    @staticmethod
    def SetTagVal(tag: str, value: Union[int, float]) -> int:
        if isinstance(value, (np.int32, np.int64)):
            value = int(value)
        if not isinstance(tag, str):
            return 0
        if not isinstance(value, (int, float)):
            return 0
        else:
            return 1

    @staticmethod
    def GetTagVal(tag: str) -> int:
        if tag == "playback":  # return 0 so wait function won't block
            return 0
        if not isinstance(tag, str):
            return 0
        return 1

    @staticmethod
    def ReadTagV(tag: str, n_start: int, n_samples: int) -> Union[int, list]:
        if not isinstance(tag, str):
            return 0
        if not isinstance(n_start, int):
            return 0
        if not isinstance(n_start, int):
            return 0
        if n_samples == 1:
            return 1
        if n_samples > 1:
            return [random.random() for i in range(n_samples)]

    @staticmethod
    def zBusTrigA(rack_num: int, trig_type: int, delay: int) -> int:
        if not isinstance(rack_num, int):
            return 0
        if not isinstance(trig_type, int):
            return 0
        if not isinstance(delay, int):
            return 0
        return 1

    @staticmethod
    def zBusTrigB(rack_num: int, trig_type: int, delay: int) -> int:
        if not isinstance(rack_num, int):
            return 0
        if not isinstance(trig_type, int):
            return 0
        if not isinstance(delay, int):
            return 0
        return 1

    class _oleobj_:
        # this is a hack and should be fixed
        @staticmethod
        def InvokeTypes(arg1, arg2, arg3, arg4, arg5, tag, arg6, value):
            return 1