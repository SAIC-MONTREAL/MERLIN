"""
Data-reformatting utilities.
"""
import copy

import numpy as np
import pickle as pkl
import copy
import collections.abc
import json

def rupdate(d, u):
    """
    Recursively update d from u
    """
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = rupdate(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def sim_obs(obs):
    """
    Convert from old (python object based) to new (numpy based) simulator spec.
    """
    out = {}
    if 'simulation_time' in obs:
        out['simulation_time'] = obs['simulation_time']

    if 'UEs' in obs:
        out['UEs'] = {}
        for k in obs['UEs'][0].keys():
            out['UEs'][k] = np.array([ue_obs[k] for ue_obs in obs['UEs']])
            if k in ['rsrp', 'rsrq', 'rssi', 'sinr']:
                out['UEs'][k] = np.transpose(out['UEs'][k])
    if 'cells' in obs:
        out['cells'] = {}
        for k in obs['cells'][0].keys():
            out['cells'][k] = np.array([cell_obs[k] for cell_obs in obs['cells']])
    return out


def reformat_evaluation_data(states, env_info):
    '''
    Format the data from evaluate.py.
    The data has same format as env states, with dimension expanded at dim 0 to hold all states at different simulation time.
    '''
    out = None
    for state in states:
        if out is None:
            out = copy.deepcopy(state)
            # Expand the dimention of np.array for later concatenation.
            for key_1, value_1 in out.items():
                if isinstance(value_1, dict):
                    for key_2, value_2 in value_1.items():
                        out[key_1][key_2] = np.expand_dims(out[key_1][key_2], axis=0)
                else:
                    out[key_1] = np.expand_dims(out[key_1], axis=0)
            continue
        for key_1, value_1 in state.items():
            if isinstance(value_1, dict):
                for key_2, value_2 in value_1.items():
                    out[key_1][key_2] = np.concatenate((out[key_1][key_2], [value_2]), axis=0)
            else:
                out[key_1] = np.concatenate((out[key_1], [value_1]), axis=0)
    out['env_info'] = env_info
    return out

def forcejson(v, **kwargs):
    """
    Serialize objects that can't be serilized using json.dump due to some
    elements being non json serilizable
    """
    if not 'default' in kwargs:
        kwargs['default'] = lambda o: str(o)
    return json.dumps(np2json(v), **kwargs)

def np2json(v):
    """
    Recursively go through object, converting any numpy values to non-numpy equivalents for
    JSON serialization
    """
    if isinstance(v, (np.int64, np.int32, np.int16, np.int8)):
        return int(v)
    if isinstance(v, (np.float, np.float32, np.float64)):
        return float(v)
    if isinstance(v, (np.ndarray, list)):
        return [np2json(x) for x in v]
    if isinstance(v, dict):
        return {np2json(k): np2json(value) for k, value in v.items()}
    if isinstance(v, tuple):
        return tuple(np2json(x) for x in v)
    return v

def str2np(s):
    x = s.replace('\n', ' ').replace('[', '').replace(']', '').split(' ')
    out = []
    for t in x:
        try:
            v = float(t)
        except ValueError:
            continue
        out.append(v)
    return np.array(out)
