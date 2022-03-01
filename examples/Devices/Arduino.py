from labplatform.config import get_config
from labplatform.core.Device import Device
from labplatform.core.Setting import DeviceSetting
from labplatform.utilities.RingBuffer import RingBuffer

import serial
import serial.tools.list_ports
from traits.api import Int, Instance, Str, Float, Any, Property
import numpy as np
import time

import logging
log = logging.getLogger(__name__)


BITSPERCOMMAND = 11
BITSPERCHECKRESP = 7
READTIMEOUT = 0.5   # read timeout


class ArduinoSetting(DeviceSetting):
    device_type = 'Arduino'

    sampling_freq = Float(100, group='primary', dsec='sampling frequency in Hz')
    buffer_time = Float(10, group='primary', dsec='length of buffer for incoming data, s')
    shape = Any((-1, 1), group='primary', dsec='shape for the data buffer')
    dtype = Any(np.float, group='primary', dsec='data type of incoming data')

    baudrate = Int(group='status', dsec='Baudrate of the Serial channel, Hz')
    Arduino_name = Str(group='status', dsec='name of the Arduino defined in the Arduino script')
    port = Str(group='status', dsec='Serial port the device is connected to')
    BytesPerCommand = Int(BITSPERCOMMAND, group='status', dsec='length of the command send to the Arduino')
    BitsCheckResp  = Int(BITSPERCHECKRESP, group='status', dsec='length of the command send to the Arduino')
    BoardDsec = Str('CH340', group='status', dsec='description of the Arduino when accessed from serial port')

    buffer_length = Property(depends_on=['buffer_time', 'sampling_freq'], group='derived',
                             dsec='buffer length in data points')

    def _get_buffer_length(self):
        return int(self.sampling_freq * self.buffer_time)


class ArduinoDevice(Device):
    serialPort = Instance(serial.Serial)
    initial_msg = Str
    buffer     = Instance(RingBuffer)
    N_         = Int
    PacketsAcquired = Int(0)

    setting = Instance(ArduinoSetting, ())

    _use_default_thread = True

    def _find_Arduino(self, read_timeout=READTIMEOUT, baudrates=None):
        if not self.setting.port and not self.setting.Arduino_name:
            raise ValueError('Need at least one of {Port name, Arduino name} to find the Arduino')
        # when port is known
        if self.setting.port:
            if not self.setting.baudrate:
                self.setting.baudrate = find_baudrate(self.setting.port, read_timeout=read_timeout, baudrates=baudrates)

            Arduino_name = find_name(self.setting.port, self.setting.baudrate, read_timeout=read_timeout)
            if not Arduino_name:
                raise ValueError('Provided baudrate {} give no response. Check the baudrate.'.
                                     format(self.setting.baudrate))
            else:
                if Arduino_name != self.setting.Arduino_name:
                    log.warning('Info provided by the device: {} does not match the software setting: {}'.
                                format(Arduino_name, self.setting.Arduino_name))
                self.setting.Arduino_name = Arduino_name
        # when only device name is known
        else:
            mpus = findArduinoSerial(self.setting.BoardDsec, self.setting.baudrate)
            for d in mpus:
                if d.name == self.setting.Arduino_name:
                    self.setting.baudrate = d.baudrate
                    self.setting.port = d.device
                    break
            if not self.setting.port:
                raise ValueError('Specified Arduino with name: {} not found'.format(self.setting.Arduino_name))

    def _configure(self, **kwargs):
        # nothing needs to be passed to the Arduino
        pass

    def _initialize(self):
        # check if all information about the board is known
        if not self.setting.port or not self.setting.Arduino_name:
            self._find_Arduino()
        # open serial port to connect to the Arduino
        try:
            self.serialPort = serial.Serial(port=self.setting.port, baudrate=self.setting.baudrate)
            # wait until the Arduino initialize
            time.sleep(1)
        except Exception as e:
            raise
        if self.serialPort.in_waiting:
            self.initial_msg = self.serialPort.read_all()
        # setup correct buffer store incoming data
        # for Mega_IO, each packets has 13 bytes, 1-8 for 8 analog channels, 9 for 8 digital channels, 10, 11 are times,
        # and 12, 13 are two packet end bytes (value 254, 255); data is uint8
        self._get_buffer()

    def _get_buffer(self):
        self.buffer = RingBuffer(shape=((self.setting.buffer_length, ) + self.setting.shape[1:]),
                                 dtype=self.setting.dtype)

    def _start(self):
        # todo
        self._sendMessage('start')

    def _pause(self):
        # todo
        pass

    def _stop(self):
        # send stop message to Arduino
        self._sendMessage('stop')
        time.sleep(0.05)
        self.serialPort.close()

    def _sendMessage(self, command, paras=0):
        """
        Interface to outgoing communication with the Arduino
        :param command: defines Type of Action to perform
        :param paras: iterable, the parameters for each command (see list in switch below for each)
        :return: None

        With these a message to the Arduino is constructed which has the following format:
        11 Chars/Uint8
        [ b {command char} {parameters} {zeros to fill to 11} ]
        e.g. "bd000000000"  for starting (=[98,100,48,48,48,48,48,48,48,48,48], as uint8)
        """
        command_len = self.setting.BytesPerCommand
        command = Arduino_command(command, paras, command_len)
        # write command to the serial port
        self.serialPort.write(command)
        log.debug('{}: Sent Command {} to Arduino'.format(self.name, command))

    def _read_serial_data(self):
        """
        read data from serial port in connection with the Arduino and write it into buffer
        Returns:
            None
        """
        raise NotImplementedError

    def thread_func(self):
        raise NotImplementedError


def findAZboards():
    '''
        find Arduino Mega equivalent 3rd party AZ mega 2560 boards connected
        we are using AZdelivery Mega2560 (a 3rd party arduino board). this program find the COM port the board connected
        to.
        device description of these boards seems to be 'USB-Serial CH340'
    :return:
        pySerial port object; use port.device to get the COM port and port.name to get the device name
    '''

    AZboards = []

    # find AZ mega boards
    for p in serial.tools.list_ports.comports():
        if 'CH340' and 'SERIAL' in p.description:
            AZboards.append(p)

    # ask for information from AZ mega boards
    # maybe should check if those ports are already opened (no remedy for the situation?)
    for bi in range(len(AZboards)):
        with serial.Serial(port=AZboards[bi].device, baudrate=get_config('ARDUINO_BAUDRATE')) as ser:
            # open serial ports reset the Arduino. need to wait for 1 second before it is running
            time.sleep(1)
            # see arduino code for the command
            ser.write(b'bi000000000')
            # wait some time before arduino gives response
            time.sleep(0.05)
            if ser.in_waiting > 0:
                try:
                    DName = ser.readline()
                    AZboards[bi].name = DName[:-1].decode('utf-8')
                    # clear serial buffer
                    str = ser.read(ser.in_waiting)
                except serial.SerialException as e:
                    print(e)
            else:
                print('not receiving response')
            # port closed with context manager

    return AZboards


def findArduinoSerial(dsec='CH340', Baudrate=None, getName=True):
    """
    find Arduino connected to the Serial port (USB connection)

    Args:
        dsec: str, description of the board when connected from the USB
            for AZdelivery Mega2560 (a 3rd party arduino board), it is 'USB-Serial CH340'
            for Arduino UNO, it is 'Arduino Uno'
        Baudrate: Int, connection speed; must match Arduino Serial configuration
        getName: Bool, if communicates with the device to get the device name
    Returns:
        pySerial port object; use port.device to get the COM port and port.name to get the device name
    """
    Arduino = []
    # find connected processors
    for p in serial.tools.list_ports.comports():
        if dsec in p.description:
            Arduino.append(p)

    if not Baudrate or Baudrate is None:
        for ar in Arduino:
            ar.baudrate = find_baudrate(ar.device)

    # ask for information from AZ mega boards
    # maybe should check if those ports are already opened (no remedy for the situation?)
    if getName:
        for bi in Arduino:
            bi.name = find_name(bi.device, bi.baudrate)

    return Arduino


def Arduino_command(command, para=0, commandLen=BITSPERCOMMAND, encoding='utf-8'):
    """
    Interface to outgoing communication with the Arduino
    Args:
        command: Defines Type of Action to perform
        para: iterable, parameters to be added for each command (see list in switch below for each)
        commandLen: Int, length in bits for each command

    Returns:
        byte string, actual command to be sent to Arduino

    With these a message to the Arduino is constructed which has the following format:
    commandLen Chars/Uint8
    [ b {command char} {parameters} {zeros to fill to 11} ]
    e.g. "bd000000000"  for starting (=[98,100,48,48,48,48,48,48,48,48,48], as uint8)
    """
    full_cmd = bytearray(b'b')
    full_cmd.extend(command.encode(encoding))
    b_para = bytearray(para)
    if b_para.__len__() <= commandLen - 2:
        b_para.extend(([0] * (commandLen - b_para.__len__() - 2)))
        full_cmd.extend(b_para)
    else:
        raise ValueError("Length of parameter array is too long (actual length {}, maximum {} allowed)".
                         format(b_para.__len__(), commandLen - 2))
    return full_cmd


def find_name(port_name, baudrate=9600, command=Arduino_command('i'), read_timeout=READTIMEOUT):
    """
    find the name of the Arduino
    Args:
        port_name: str, the name of the serial port the device connected to
        baudrate: int, baudrate of the serial port
        command: byte array, command used to communicate with the device, see Arduino_Command
        read_timeout: float, timeout in second for the testing

    Returns:
        str, name of the Arduino
    """
    with serial.Serial(port=port_name, baudrate=baudrate) as ser:
        # open serial ports reset the Arduino. need to wait for 1 second before it is running
        time.sleep(1)
        ser.timeout = read_timeout
        # see arduino code for the command
        if ser.in_waiting:
            ser.read_all()
        ser.write(command)
        # wait some time before arduino gives response
        time.sleep(0.05)
        try:
            resp = ser.read(ser.in_waiting)
            if resp:
                name = resp.decode('utf-8').strip('\n')
                return name
            else:
                print('not receiving response')
                # port closed with context manager
        except serial.SerialException as e:
            print(e)


def find_baudrate(port_name, command=Arduino_command('r'), resp_length=BITSPERCHECKRESP,
                  read_timeout=READTIMEOUT, baudrates=None):
    """
    find the baudrate used by the connected device
    Args:
        port_name: str, the name of the serial port the device connected to
        command: byte array, command used to communicate with the device, see Arduino_Command
        resp_length: int, length of expected response
        read_timeout: float, timeout in second for the testing
        baudrates: list or tuple, baudrates to be tested. if None use serial.Serial.BAUDRATES, and minimum is 9600

    Returns:
        int, baudrate used by the device
    """
    if not baudrates:
        baudrates = np.array(serial.Serial.BAUDRATES)
        baudrates = baudrates[baudrates > 9599]

    with serial.Serial(port=port_name) as ser:
        time.sleep(1)
        ser.timeout = read_timeout
        ser.read_all()  # clear read buffer
        for br in baudrates:
            ser.baudrate = br
            ser.write(command)
            resp = ser.read(resp_length)
            if resp == b'brcheck':
                # clear read buffer
                ser.read_all()
                return br
    return None
