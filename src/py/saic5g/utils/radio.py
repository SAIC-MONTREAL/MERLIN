"""
Radio-related utilities
"""
import numpy as np

def to_db(val):
    return 10 * np.log10(val)

def from_db(val):
    return np.power(10, val / 10.)
