from pyvcam import constants as camConst
import ctypes


def get_camera_params(cam, valid_only=True):
    """
    for each defined parameters in pyvcam.constants, open each camera and find:
        1) if this parameter is available in this camera
        2) what are the possible values
        3) what is the current value

    Args:
        cam: camera object returned by pyvcam.Camera
        valid_only: if True, omit parameters not application/cannot be read from the camera

    Returns:
        a dictionary
    """
    if not cam.is_open:
        cam.open()

    params = dict()
    # all definded parameters
    for key in sorted(camConst.__dict__.keys()):
        # seems all parameters starts with pattern 'PARAM_'
        is_valid = False
        if key.startswith('PARAM_'):
            par = dict()
            # query the camera
            try:
                cam.get_param(getattr(camConst, key), camConst.ATTR_AVAIL)
                try:
                    par['value_available'] = cam.read_enum(getattr(camConst, key))
                except RuntimeError:
                    par['value_available'] = {}
                if key is not 'PARAM_DD_INFO':
                    try:
                        par['value_current'] = cam.get_param(getattr(camConst, key), camConst.ATTR_CURRENT)
                    except RuntimeError:
                        par['value_current'] = 'N/A'
                else:
                    par['value_current'] = '? resulting in error'
                try:
                    par['accessible'] = cam.get_param(getattr(camConst, key), camConst.ATTR_ACCESS)
                except RuntimeError:
                    par['accessible'] = 'N/A'
                is_valid = True
            except AttributeError:
                par['value_available'] = {'error': 'attribute'}
                par['value_current'] = 'N/A'
                par['accessible'] = 'N/A'
            except RuntimeError:
                par['value_available'] = {}
                par['value_current'] = 'unKnown: datatype not match'
                par['accessible'] = 'unKnown: datatype not match'

            if (valid_only and is_valid) or not valid_only:
                params[key] = par

    return params


def pp_camera_params(params, cmd_print=True):
    """
    pretty-printing the list of parameters to the command line

    Args:
        params: dictionary returned by function get_camera_params
        cmd_print: if print to command line or not

    Returns:
        fmt_params: list of tuples, formatted parameters with information
        col_paddings: list of int, the padding space between each column
    """

    # a list of formatted parameters; each element is a tuple for one line, the tuple should have 3 elements:
    # name + available values + current value
    fmt_params = []
    fmt_params.insert(0, ('param_name', 'accessible', 'available_values', 'current_value'))
    fmt_params.insert(1, ('------------', '-----------', '-----------------', '---------------'))

    for key in params:
        # check if the params contains unprintable characters
        nd = params[key].copy()
        for ndkey in nd.keys():
            if not str(nd[ndkey]).isprintable():
                nd[ndkey] = repr(nd[ndkey])
        if nd['value_available'].__len__() > 0:
            for i, pk in enumerate(nd['value_available'].keys()):
                if i == 0:
                    fmt_params.append((key, nd['accessible'],
                                       pk + ': ' + str(nd['value_available'][pk]),
                                       nd['value_current']))
                else:
                    fmt_params.append((' ', ' ',
                                       pk + ': ' + str(nd['value_available'][pk]), ' '))
        else:
            fmt_params.append((key, nd['accessible'], 'NA', nd['value_current']))

    # calculate column padding
    col_paddings = []
    for i in range(len(fmt_params[0])):
        sizes = [len(str(row[i])) if row[i] is not None else 0 for row in fmt_params]
        col_paddings.append(max(sizes))

    # print the formated list to command line
    if cmd_print:
        print('\n')
        for i, row in enumerate(fmt_params):
            msg = ' ' * 4
            for j, col in enumerate(row):
                msg += str(col).ljust(col_paddings[j] + 4)
            print(msg)

    return fmt_params, col_paddings


def print_params_to_file(fmt_params, padding, cam_name='PrimeCam', user_tag='default', overwrite=False):
    """
    writes formatted camera parameters to a txt file; file name follows CamName_UserTag_Date_index
    Args:
        fmt_params: formatted parameters returned by function pp_camera_params to be written
        padding:
        cam_name: str, name of the camera
        user_tag: str, added to file name
        overwrite: if overwrite existing file or not; if True, always overwrite last written file

    Returns:
        None
    """
    from labplatform.config import get_config
    import os
    from datetime import datetime
    f_dir = os.path.join(get_config('DEVICE_ROOT'), cam_name)
    date = datetime.now().date().strftime(get_config('DATE_FMT'))
    fname = cam_name + '_' + user_tag + '_' + date
    # check if file with same name already exist
    all_files = [s for s in os.listdir(f_dir)
                 if os.path.isfile(os.path.join(f_dir, s)) and s.startswith(fname)]
    if not all_files:
        fname = fname + '_' + '0'.rjust(get_config('FILL_DIGIT'), '0')
    else:
        all_files = sorted(all_files)
        if overwrite:
            fname = all_files[-1]
        else:
            last_digit = int(all_files[-1].split('_')[-1]) + 1
            fname = fname + '_' + str(last_digit).rjust(get_config('FILL_DIGIT'), '0')

    # write to file
    with open(os.path.join(f_dir, fname), 'w') as fh:
        for i, row in enumerate(fmt_params):
            msg = ' ' * 4
            for j, col in enumerate(row):
                msg += str(col).ljust(padding[j] + 4)
            fh.write(msg + '\n')


def create_SMART_struct(smart_sequence):
    """
    create a ctype structure to be used to control the SMART streaming function of the camera

    Args:
        smart_sequence: list, tuple or np.array; smart sequency
    Returns:
        instance of ctypes.Structure
    """

    entries = ctypes.c_uint16
    params = ctypes.c_uint32 * len(smart_sequence)

    sc = type('smart_struct', (ctypes.Structure, ), {'_fields_': [('entries', entries),
                                                                  ('params', params)]})

    smart_struct = sc()
    smart_struct.entries = len(smart_sequence)
    smart_struct.params = params(*smart_sequence)

    return smart_struct
