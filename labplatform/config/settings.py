"""
    environment settings used by the Lab platform

"""

import os, sys, logging
import numpy as np

# defined experiments
EXPERIMENTER = ['Chao', 'Marc']
PARADIGM = ['Test', 'Intrinsic_Imaging', 'Calcium_Imaging']
# ANIMALS = [1, 2, 3, 4, 5]

COMPUTERNAME = os.environ['COMPUTERNAME']
CODE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

# file system
import warnings
import textwrap
# Default to the user's home directory and raise a warning.
if sys.platform == 'win32':
    BASE_DIRECTORY = os.path.join('C:', os.path.sep, 'labplatform')
else:
    BASE_DIRECTORY = os.path.join(os.path.expanduser('~'), 'labplatform')
mesg = '''
    By default, use computer's base directory, {}, to hold all the relevant files. '\n The paradigm settings,
    calibration data, log files and data files are stored here. '''
mesg = textwrap.dedent(mesg.format(BASE_DIRECTORY))
# mesg = mesg.replace('\n', ' ')
warnings.warn(mesg)
# print(mesg)

LOG_ROOT        = os.path.join(BASE_DIRECTORY, 'logs')        # log files
DATA_ROOT       = os.path.join(BASE_DIRECTORY, 'data')        # data files
SUBJECT_ROOT    = os.path.join(BASE_DIRECTORY, 'subjects')  # subjects files
CAL_ROOT        = os.path.join(BASE_DIRECTORY, 'calibration')  # calibration files
SETTINGS_ROOT   = os.path.join(BASE_DIRECTORY, 'settings')
SOUND_ROOT      = os.path.join(BASE_DIRECTORY, 'sound_files')  # sound data
TEMP_ROOT       = os.path.join(BASE_DIRECTORY, 'temp')        # temp files
PARADIGMS_ROOT  = os.path.join(BASE_DIRECTORY, 'paradigms')
DEVICE_ROOT     = os.path.join(BASE_DIRECTORY, 'Devices')   # device controlling files
SERVER_ROOT     = ''

SUB_ROOTS = {'LOG_ROOT':        LOG_ROOT,
             'DATA_ROOT':       DATA_ROOT,
             'SETTING_ROOT':    SETTINGS_ROOT,
             'TEMP_ROOT':       TEMP_ROOT,
             'DEVICE_ROOT':     DEVICE_ROOT,
             'SOUND_ROOT':      SOUND_ROOT,
             'CAL_ROOT':        CAL_ROOT,
             'PARADIGM_ROOT':   PARADIGMS_ROOT,
             'SUBJECT_ROOT':    SUBJECT_ROOT,
             'SERVER_ROOT':     ''}


# check to see if the dirs exist; create dir if it does not exist
if not os.path.isdir(BASE_DIRECTORY):
    logging.info('Creating base directory, {}'.format(BASE_DIRECTORY))
    os.mkdir(BASE_DIRECTORY)
    for p in SUB_ROOTS:
        if not os.path.isdir(SUB_ROOTS[p]) and SUB_ROOTS[p]:
            logging.info('Creating directory at {}'.format(SUB_ROOTS[p]))
            os.mkdir(SUB_ROOTS[p])


PARADIGMS_SUB_ROOTS = {}
for exp in PARADIGM:
    PARADIGMS_SUB_ROOTS[exp+'_ROOT']    = os.path.join(PARADIGMS_ROOT, exp)
    PARADIGMS_SUB_ROOTS[exp+'_DATA']    = os.path.join(PARADIGMS_SUB_ROOTS[exp+'_ROOT'], 'data')
    PARADIGMS_SUB_ROOTS[exp+'_LOG']     = os.path.join(PARADIGMS_SUB_ROOTS[exp+'_ROOT'], 'log')
    PARADIGMS_SUB_ROOTS[exp+'_SETTING'] = os.path.join(PARADIGMS_SUB_ROOTS[exp+'_ROOT'], 'setting')
    PARADIGMS_SUB_ROOTS[exp+'_TEMP'] = os.path.join(PARADIGMS_SUB_ROOTS[exp+'_ROOT'], 'temp')


# Default filename extensions used by the FileBrowser dialog to open/save files.
COHORT_WILDCARD     = 'Cohort files (*.cohort.hd5)|*.cohort.hd5|'
PARADIGM_WILDCARD   = 'Paradigm settings (*.par)|*.par|'
PHYSIOLOGY_WILDCARD = 'Physiology settings (*.phy)|*.phy|'

# Separator used by pytables
PYTABLES_SEP = '/'
# limit for HDF5 metadata size for single parameter
PYTABLES_ATTR_SIZELIMIT = 1e4   # byte

# Format to use when generating time strings for use in a HDF5 node pathname
# (see time.strptime for documentation re the format specifiers to use below)
TIME_FORMAT = '%Y-%m-%d-%H-%M-%S'

# Format to use when storing datetime strings as attributes in the HDF5 file
DATE_FMT = '%Y-%m-%d'
TIME_FMT = '%H-%M-%S'
DATETIME_FMT = DATE_FMT + ' ' + TIME_FMT

# Digits to left fill when creating experiment and trial names in the data file
FILL_DIGIT = 4

CONTROL_INTERVAL = 0.1  # second
MAX_ANALOGOUT_VOLTAGE = 10  # volt
MAX_ANALOGIN_VOLTAGE = 10  # volt
MIN_ANALOGOUT_VOLTAGE = 0  # volt
MIN_ANALOGIN_VOLTAGE = 0  # volt
CHUNK_SIZE = 50e6  # dimensionless
# Size of sample (in seconds) to use for computing the noise floor
NOISE_DURATION = 15  # second
# By convention, settings are in all caps.  Print these to the log file to
# facilitate debugging other users' programs.
log = logging.getLogger()
for k, v in sorted(globals().items()):
    if k == k.upper():
        log.debug("PLATFORM SETTING %s : %r", k, v)

# load subject information

# timeout for initializing thread/process
TIMEOUT_PROCESS = 30   # second

'''
# Arduino parameters
ARDUINO_BAUDRATE = 115200
ARDUINO_BytesPerPacket = 13
ARDUINO_CONTROLINTERVAL = 0.05      # in second
ARDUINO_TYPE = {'Mega_IO': 'ArduinoMega'}
ARDUINO_PARA = {'Mega_IO':{'BaudRate':ARDUINO_BAUDRATE, 'ControlInterval':ARDUINO_CONTROLINTERVAL, 'NAnalogIN':8,
                           'NDigitalIN':8, 'StepSize':1000, 'InputBufferSize':100*ARDUINO_BAUDRATE,
                           'BytesAvailableFcnMode':'byte', 'TimeInd':ARDUINO_BytesPerPacket - np.array([3,2]),
                           'BusyInd':ARDUINO_BytesPerPacket-4, 'BytesPerPacket':ARDUINO_BytesPerPacket,
                           'BytesPerCommand':11, 'PacketsDisplayed':1000, 'StopCharAsInt':np.array([254,255]),
                           'AnalogRange':np.array([0,5]), 'DigitalRange':np.array([0,1]),'AbortPin':38
                           }
                }
'''

