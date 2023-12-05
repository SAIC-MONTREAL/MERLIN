import os
from hydra.utils import get_original_cwd

def make_tuple(*args):
    """
    Helper function for instantiating tuples from hydra config files

    The positional arguments will be put into the tuple.
    """
    return tuple(args)

def abspath(path):
    """
    Hydra tampers with the current working directory, which makes
    it difficult to specify relative paths in the hydra config.
    This function takes a relative path and changes it to absolute,
    using the original present working directory. If the given path is
    already absolute, then we just return the path as is.

    See the following links:

    https://github.com/facebookresearch/hydra/issues/910
    https://github.com/facebookresearch/hydra/issues/1651

    Args:
        path (str): The path we would like to convert.
    """

    if not os.path.isabs(path):
        return os.path.join(get_original_cwd(), path)
    return path

def get_one(cfg, max_selection=None):
    """Get one item from a configuration dictionary

    Hydra supports the selection of multiple items from a config group.
    Sometimes, we would like to ensure that only a single item is selected.
    At the same time, we would like to get the single item under the config group.
    This function helps to extract one item from cfg, and can check that
    no more than max_selection items is directly under cfg. If max_selection
    is None, then no check is performed.

    For example, if::

        cfg = {
            sls: {
                a: 1,
                b: 2
            }
        }

    Then get_one(cfg, max_selection=1) will return the tuple (sls, {a:1, b:2}).
    Additionally, we will ensure that sls is the only item inside cfg.

    Args:
        cfg (dict): The configuration dictionary from which to get an item.
    """
    if max_selection is not None:
        assert len(cfg) <= max_selection, 'Not expecting multiple config selections: %s' % str(list(cfg.keys()))
    first_key = list(cfg.keys())[0]
    return first_key, cfg[first_key]
