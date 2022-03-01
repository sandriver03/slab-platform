import tables
import os
import logging
import sys
import numpy as np

log = logging.getLogger(__name__)

name_lookup = {'group': 'group',
               'earray': 'earray',
               'array': 'array',
               'table': 'table'}


if sys.version.startswith('3'):
    unicode = str


# used to save a nested dictionary to a pytables.Table node
def get_dict_dtype(data):
    """Given a dict, generate a nested numpy dtype"""
    fields = []
    for (key, value) in data.items():
        # make strings go to the next 64 character boundary
        # pytables requires an 8 character boundary
        if isinstance(value, unicode):
            value += u' ' * (64 - (len(value) % 64))
            # pytables does not support unicode
            if isinstance(value, unicode):
                value = value.encode('utf-8')
        elif isinstance(value, str):
            value += ' ' * (64 - (len(value) % 64))

        if isinstance(value, dict):
            fields.append((key, get_dict_dtype(value)))
        else:
            value = np.array(value)
            fields.append((key, '%s%s' % (value.shape, value.dtype)))
    return np.dtype(fields)


def _dict_recurse_row(row, base, data):
    for (key, value) in data.items():
        new = base + key
        if isinstance(value, dict):
            _dict_recurse_row(row, new + '/', value)
        else:
            row[new] = value


def _recurse_tbl2dict(descr, data):
    res = dict()
    for key, item in zip(descr, data):
        if isinstance(key, (list, tuple)) and len(key) > 1:   # a nested description
            res[key[0]] = _recurse_tbl2dict(key[1], item)
        else:
            res[key] = item
    return res


def dict_add_row(tbl, data):
    """
    Add a new row to a table based on the contents of a dict.
    """
    row = tbl.row
    for (key, value) in data.items():
        if isinstance(value, dict):
            _dict_recurse_row(row, key + '/', value)
        else:
            row[key] = value
    row.append()
    tbl.flush()


def dict_read_row(tbl, start=None, stop=None, step=None):
    """
    read selected rows from the table, and return as a dictionary
    Args: see tables.Table.read

    Returns:
        list of dict
    """
    data = tbl.read(start, stop, step)
    res = []
    # format data into dictionary, using information from description
    for row in data:
        res.append(_recurse_tbl2dict(tbl.description._v_nested_names, row))
    return res


# for some reason the tables.file._open_files.get_handlers_by_name is not working
# create a method which performs similar function
def get_opened_files():
    # return a list of opened files, by name
    fh = tables.file._open_files.handlers
    return [h.filename for h in fh]


def get_handler_by_name(fname):
    """
    return a file handler pointed to file defined by fname (a string contains full path to the file). If more than 1
    handles to the same file is found, raise a warning
    Args:
        fname: str

    Returns:
        handle to the file if the file is opened, otherwise none
    """
    if not isinstance(fname, str):
        raise ValueError('parameter fname must be str, but {} is provided'.format(fname.__class__))

    handler_list = [h for h in tables.file._open_files.handlers]
    handler_name = [h.filename for h in handler_list]
    if fname in handler_name:
        if handler_name.count(fname) > 1:
            raise Warning('More than 1 handlers found for file: {}'.format(fname))
        return handler_list[handler_name.index(fname)]
    else:
        return None


def get_or_append_node(node, name, type='group', *arg, **kw):
    try:
        return getattr(node, name)
    except tables.NoSuchNodeError:
        return append_node(node, name, type, *arg, **kw)


def append_node(node, name, type='group', *arg, **kw):
    log.debug('appending {} to node {}'.format(name, node._v_pathname))
    type = name_lookup[type.lower()]
    func = getattr(node._v_file, 'create_' + type)
    return func(node, name, *arg, **kw)


def update_table(table, idx=None, close_on_finish=True, **kwargs):
    """
    Update a tables.Table node with information provided in kwargs.

    Args:
        table: a handle to the Table node to be updated
        idx: index of the row to be updated. if None, create a new row
        close_on_finish: if close the file after finish. by default is True
        **kwargs: key and value pairs defined by the description of the Table node

    Returns:
        None
    """
    if idx is not None:  # record found, update its content
        for k, v in kwargs.items():
            table.modify_column(idx, colname=k, column=[v])
    else:  # record not found, create a new entry
        r = table.row
        for k, v in kwargs.items():
            r[k] = v
        r.append()
        r.table.flush()

    if close_on_finish:
        table._v_file.close()


def find_table_index(table, expr, opts):
    """
    Find the index of an Table entry (row) defined in opts, in the Table node provided by table

    Args:
        table: tables.Table handle
        expr: an expression which can be used in table.where() method. see pytables.Table.where for more information
        opts: dictionary defines which value to search for

    Returns:
        an index if found, or None
    """
    index = [row.nrow for row in table.where(expr, opts)]
    if len(index) == 1:
        return index[0]
    elif len(index) == 0:
        log.warning('Cannot find the entry for current experiment: {} in the record'
                    .format(opts['exp_name']))
        return None
    else:
        raise ValueError('More than one entry for current experiment: {} were found'
                         .format(opts['exp_name']))


def get_file_handle(data_file_path, mode='a'):
    """
    Get the handle to the HDF5 file
    Args:
        data_file_path: str, absolute path to the file
        mode: str, file opening mode
    Returns:
        tables.File
    """
    if not data_file_path:
        raise ValueError('Data file not known')
    if get_handler_by_name(data_file_path):
        h = get_handler_by_name(data_file_path)
    else:
        # first check if the folder is present
        if not os.path.isdir(os.path.split(data_file_path)[0]):
            os.makedirs(os.path.split(data_file_path)[0])
        try:
            h = tables.open_file(data_file_path, mode=mode)
        except:
            raise
    return h


def get_node_handle(file_path, node_path, mode='a'):
    """
    get the handle to a specific node in a HDF5 file. the file is kept open
    Args:
        file_path: str, absolute path to the file
        node_path: str, path of the node inside the file
        mode: str, mode used to open the file

    Returns:
        tables.Group
    """
    h = get_file_handle(file_path, mode)
    if isinstance(node_path, str) and node_path[0] != '/':
        node_path = '/' + node_path
    return h.get_node(node_path)


def _parse_xattr(xattr):
    xattr = xattr.replace('../', '/_v_parent/')  # replace ../ with _v_parent
    xattr = xattr.replace('+', '/_v_attrs/')  # replace + with _v_attrs
    xattr = xattr.replace('//', '/')  # remove double slashes
    # xattr = xattr.replace('/', '.')             # convert to attribute access
    xattr = xattr.replace('</', '<')
    xattr = xattr.strip('/')  # remove leading period
    return xattr


def _getattr(node, xattr):
    if xattr.startswith('<'):
        return _find_ancestor(node, xattr[1:])
    elif xattr == '*':
        return node._f_list_nodes()[0]
    elif xattr == '_v_parent':
        return node._v_parent
    else:
        # Try to get the child node named xattr.  If it fails, see if the
        # regular getattr method works (e.g. special attributes such as
        # _v_pathname and _v_name can only be accessed via getattr).
        try:
            # If the node is an instance of tables.Leaf, then it will not have
            # the _f_getChild method and raises an AttributeError.  If the node
            # is an instance of tables.Group, then it will have the _f_getChild
            # method and raise a tables.NoSuchNodeError (a subclass of
            # AttributeError) if the node does not exist.  We can capture both
            # and fall back to the getattr approach.
            return node._f_get_child(xattr)
        except AttributeError:
            return getattr(node, xattr)


def _rgetattr(node, xattr):
    try:
        base, xattr = xattr.split('/', 1)
        # Handle the special cases first
        if base == '_v_attrs':
            return node._f_getattr(xattr)
        else:
            node = _getattr(node, base)
            return _rgetattr(node, xattr)
    except ValueError:
        # This means we are on the very last attribute to be fetched (i.e.
        # there's no more '.' in the xattr to parse.
        return _getattr(node, xattr)


def _find_ancestor(obj, xattr):
    if obj == obj._v_parent:
        raise AttributeError("{} not found".format(xattr))
    try:
        return _rgetattr(obj._v_parent, xattr)
    except AttributeError:
        return _find_ancestor(obj._v_parent, xattr)


def rgetattr(node, xattr):
    '''
    Recursive extended getattr that works with the PyTables HDF5 hierarchy::

        ../
            Move to the parent
        *
            First child node
        /name
            Move to the child specified by name
        <ancestor_name
            Find the nearest ancestor whose name matches ancestor_name
        <+attribute
            Find the nearest ancestor that has the specified attribute
        +attribute
            Get the value of the attribute

        _v_name
            Name of the current node
        ../_v_name
            Name of the parent node
        ../../_v_name
            Name of the grandparent node
        paradigm+bandwidth
            Value of the bandwidth attribute on the child node named 'paradigm'
        ../paradigm+bandwidth
            Value of the bandwidth attribute on the sibling node named 'paradigm'
            (i.e. check to see if the parent of the current node has a child node
            named 'paradigm' and get the value of bandwidth attribute on this child
            node).

    Given the following HDF5 hierarchy::

        Animal_0
            _v_attrs
                sex = F
                nyu_id = 132014
            Experiments
                Experiment_1
                    _v_attrs
                        start_time = August 8, 2011 11:57pm
                        duration = 1 hour, 32 seconds
                    paradigm
                        _v_attrs
                            bandwidth = 5000
                            center_frequency = 2500
                            level = 60 (a)
                    data
                        trial_log
                        contact_data

    You can expect the following behavior::

#        >>> node = root.Animal_0.Experiments.Experiment_1.data.trial_log
#        >>> xgetattr(node, '../_v_name')
        data
#        >>> xgetattr(node, '..')._v_name
        data
#        >>> xgetattr(node, '../..')._v_name
        Experiment_1
#        >>> xgetattr(node, '../paradigm/+bandwidth')
        5000
#        >>> xgetattr(node, '<+nyu_id')
        132014

    '''
    parsed_xattr = _parse_xattr(xattr)
    log.debug('Converted %s to %s for parsing', xattr, parsed_xattr)
    if parsed_xattr.startswith('<'):
        return _find_ancestor(node, parsed_xattr[1:])
    return _rgetattr(node, parsed_xattr)


def rgetattr_or_none(obj, xattr):
    '''
    Attempt to load the value of xattr, returning None if the attribute does not
    exist.
    '''
    try:
        return rgetattr(obj, xattr)
    except AttributeError:
        return None


def node_match(n, filter):
    '''
    Checks for match against each keyword.  If an attribute is missing or any
    match fails, returns False.

    Filter can be a dictionary or list of tuples.  If the order in which the
    filters are applied is important, then provide a list of tuples.
    '''
    # If filter is a dictionary, convert it to a sequence of tuples
    if type(filter) == type({}):
        filter = tuple((k, v) for k, v in filter.items())

    # If user only provided one filter rather than a sequence of filters,
    # convert it to a sequence of length 1 so the following loop can handle it
    # better
    if len(filter[0]) == 1:
        filter = (filter,)

    for xattr, criterion in filter:
        try:
            value = rgetattr_or_none(n, xattr)
            if not criterion(value):
                return False
        except AttributeError:
            return False
        except TypeError:
            if not (value == criterion):
                return False
    return True


def iter_nodes(where, filter, classname=None):
    '''
    Non-recursive version of func:`walk_nodes` that only inspects the immediate
    children of the node specified by `where`.

    This is an iterator.  To obtain a list (this is standard Python-fu)::

        list(iter_nodes(fh.root, ('_v_name', 'trial_log)))
    '''
    for node in where._f_iter_nodes(classname=classname):
        if node_match(node, filter):
            yield node


def walk_nodes(where, filter, classname=None):
    '''
    Starting at the specifide node, `walk_nodes` visits each node in the
    hierarchy below `where`, returning a list of all nodes that match the filter
    criteria.  This is a recursive function that visits children, grandchildren,
    great-grandchildren, etc., nodes.  For a non-recursive function that only
    inspects the immediate children see :func:`iter_nodes`.

    Filters are specified as a sequence of tuples, (attribute, filter).  As
    each node is visited, each filter is called with the corresponding value of
    its attribute.  The node is discarded if it is missing one or more of the
    filter attributes.

    Each filter may be a callable that returns a value or raises an exception.
    If the filter raises an exception or returns an object whose truth value
    is Fales, the node is discarded.

    Attributes may be specified relative to the nodes of interest using the '/'
    POSIX-style path separator.

    Attribute examples::

        _v_name
            Name of the current node
        ../_v_name
            Name of the parent node
        ../../_v_name
            Name of the grandparent node
        paradigm/+bandwidth
            Value of the bandwidth attribute on the child node named 'paradigm'
        ../paradigm/+bandwidth
            Value of the bandwidth attribute on the sibling node named
            'paradigm' (i.e. check to see if the parent of the current node has
            a child node named 'paradigm' and get the value of bandwidth
            attribute on this child node).

    If you have ever worked with pathnames via the command line, you may
    recognize that the path separators work in an identical fashion.

    Filter examples::

        ('_v_name', re.compile('[dD]data').match)
            Matches all nodes named 'Data' or 'data'

        ('_v_name', re.compile('^\d+.*').match)
            Matches all nodes whose name begins with a sequence of numbers

        ('_v_name', 'par_info')
            Matches all nodes whose name is exactly 'par_info'

        ('../../+start_time', lambda x: (strptime(x).date()-date.today()).days <= 7)
            Matches all nodes whose grandparent (two levels up) contains an
            attribute, start_time, that evaluates to a date that is within the
            last week.  Useful for restricting your analysis to data collected
            recently.

    Useful node attributes::

        _v_name
            Name of the node
        _v_pathname
            HDF5 pathname of the node
        _v_depth
            Depth of the node relative to the root node (root node depth is 0)

    If all of the attributes are found for the given node and the attribute
    values meet the filter criterion, the node is added to the list.

    To return all nodes that store animal data (note that the iterator approach
    is standard Python-fu)::

        fh = tables.openFile('example_data.h5', 'r')
        filter = ('+animal_id', lambda x: True)
        iterator = walk_nodes(fh.root, filter)
        animal_nodes = list(iterator)

    If you want to walk over the results one at a time, the above can be
    rewritten::

        fh = tables.openFile('example_data.h5', 'r')
        filter = ('+animal_id', lambda x: True)
        for animal_node in walk_nodes(fh.root, filter):
            # Do something with the node, e.g.:
            process_node(animal_node)

    To return all nodes who have a subnode, data, that has a name beginning with
    'RawAversiveData'::

        fh = tables.openFile('example_data.h5', 'r')
        base_node = fh.root.Cohort_0.animals.Animal_0.experiments
        filter = ('data._v_name', re.compile('RawAversiveData.*').match)
        experiment_nodes = list(walk_nodes(base_node, filter))

    To return all nodes whose name matches a given pattern::

        fh = tables.openFile('example_data.h5', 'r')
        filter = ('_v_name', re.compile('^Animal_\d+').match)
        animal_nodes = list(walk_nodes(fh.root, filter))
    '''
    for node in where._f_walknodes(classname=classname):
        if node_match(node, filter):
            yield node
