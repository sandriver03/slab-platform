# note about python import: it seems the code in __init__.py only runs when use import for the first time. after that,
# further imports does not run the code again. reload the module also re-run the code

import logging
import os
import getpass


log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


def _get_config_file_name(env_name, comp_name=None, usr_name=None):
    """
    load the setting configuration from a saved setting file
    first the default settings are loaded, and then the settings specific to the setup are loaded and overwrite the
    default ones if applicable
    the function modifies the globals in this module, _settings and __changed_settings
    Args:
        env_name: str, name of the setup environment
        comp_name: str, name of the computer
        usr_name: str, name of the user
    Returns:
        None
    """
    if comp_name is None:
        comp_name = os.environ['COMPUTERNAME']
    if usr_name is None:
        usr_name = getpass.getuser()

    return '{}_{}_{}.json'.format(comp_name, usr_name, env_name)


def load_settings(env_path=None):
    """
    first load the defaults, and then update the defaults with settings saved in env_path
    env_path should be an absolute path pointing to a json file
    """
    # Load the default settings
    from . import settings

    # Load the computer-specific settings, saved as separate file in Config, with name [CompName_config]
    if env_path is None:
        settings_dir = getattr(settings, 'SETTINGS_ROOT')
        conf_name = _get_config_file_name(env_name='general')
        file_abs_path = os.path.join(settings_dir, conf_name)
    else:
        if not os.path.isabs(env_path):
            raise ValueError('the path to the configuration file has to be an absolute path')
        file_abs_path = env_path

    # computer and setup specific settings should be saved with json
    extra_settings = {}
    if os.path.isfile(file_abs_path):
        try:
            import json
            with open(file_abs_path, 'r') as fh:
                # loaded data is a dict
                extra_settings = json.load(fh)

                for k, v in extra_settings.items():
                    setattr(settings, k, v)
        except IOError:
            log.debug('config file {} cannot be read', file_abs_path)
    else:
        log.debug('No Computer specific general settings defined, use default settings')

    # remove those builtin fields
    setting_names = [s for s in dir(settings) if s.upper() == s or s == '__spec__']
    setting_values = [getattr(settings, s) for s in setting_names]
    return dict(zip(setting_names, setting_values)), extra_settings


def set_config(setting, value):
    """
    Set value of a setting. alphabetic characters in setting must be all uppercase
    """
    # the value of SETTINGS_ROOT should never be changed
    if setting == 'SETTINGS_ROOT':
        raise ValueError('the value "SETTINGS_ROOT" cannot be modified')
    if setting.upper() != setting:
        log.warning('alphabetic characters in setting must be all uppercase. it has been converted')
        setting = setting.upper()
    _settings[setting] = value
    __changed_settings[setting] = value


def get_config(setting=None):
    """
    Get value of setting
    """
    if setting is not None:
        return _settings[setting]
    else:
        return get_settings()


def get_settings():
    """
    Get all current settings
    """
    return _settings


def save_env_config(env_name='general'):
    """
    save custom environment settings to the SETTING_ROOT. only those different from defaults are saved
    the changed settings can only be registered if it is done through set_config
    the file name will be in form of {comp_name}_{usr_name}_{env_name}.json
    Args:
        env_name: str, name of the setup environment
    Returns:
        None
    """
    if not __changed_settings:
        raise ValueError('No new settings to be saved')

    file_name = _get_config_file_name(env_name=env_name)
    path = get_config('SETTINGS_ROOT')
    # save settings in a text file with json
    import json
    with open(os.path.join(path, file_name), 'w') as fh:
        json.dump(__changed_settings, fh)


def load_env_config(env_name, comp_name=None, usr_name=None):
    """
    load the setting configuration from a saved setting file
    first the default settings are loaded, and then the settings specific to the setup are loaded and overwrite the
    default ones if applicable
    the function modifies the globals in this module, _settings and __changed_settings
    Args:
        env_name: str, name of the setup
        comp_name: str, name of the computer
        usr_name: str, name of the user
    Returns:
        None
    """
    global _settings, __changed_settings
    custom_set_name = _get_config_file_name(env_name, comp_name, usr_name)
    settings_dir = get_config('SETTINGS_ROOT')
    file_abs_path = os.path.join(settings_dir, custom_set_name)

    if not os.path.isfile(file_abs_path):
        raise ValueError('specified configuration file {} does not exist'.format(file_abs_path))

    # override current settings
    _settings, __changed_settings = load_settings(file_abs_path)

    # create file system if needed
    _create_dirs()


def _create_dirs():
    """
    create file system, if it is not there
    """
    # check to see if the dirs exist; create dir if it does not exist
    if not os.path.isdir(get_config('BASE_DIRECTORY')):
        log.info('Creating base directory, {}'.format(get_config('BASE_DIRECTORY')))
        os.mkdir(get_config('BASE_DIRECTORY'))

    _dict = get_config('SUB_ROOTS')
    for p in get_config('SUB_ROOTS'):
        if not os.path.isdir(_dict[p]) and _dict[p]:
            log.info('Creating directory at {}'.format(_dict[p]))
            os.mkdir(_dict[p])


# when first time importing, load the defaults, and {comp}_{usr}_{general}.json
# __changed_settings holds all pairs that are different from default
_settings, __changed_settings = load_settings()
# create file system if needed
_create_dirs()
