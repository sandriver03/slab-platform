"""
system configurations used in speaker calibration
"""
IMAGING = {'ZBus': {'use_ZBus': False, 'connection': 'GB'},
           'devices': {'RX6': {'processor': 'RX6', 'index': 1, 'connection': 'GB',
                               'rcx_file': 'RCX\\RX6_puretone.rcx',}
                       },
           'trigger': {'device': 'RX6', 'trig': 1, 'ZBus': False}
           }

TESTING = {'ZBus': {'use_ZBus': False, 'connection': 'USB'},
           'devices': {'RP2': {'processor': 'RP2', 'index': 1, 'connection': 'USB',
                                 'rcx_file': 'RCX\\RP2_puretone.rcx',}
                       },
           'trigger': {'device': 'RP2', 'trig': 1, 'ZBus': False}
           }

