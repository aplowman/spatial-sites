"""`spatial_sites.utils.py`

Module containing miscellaneous utility functions.

"""

import fractions

import numpy as np


def cast_arr_to_float(arr):
    """Cast elements of an Numpy array to floats using the fractions module."""

    float_arr = np.zeros_like(arr, dtype=float) * np.nan
    for idx, i in np.ndenumerate(arr):
        float_arr[idx] = float(fractions.Fraction(i))

    return float_arr


def check_indices(seq, seq_idx):
    """
    Given a sequence (e.g. list, tuple, ndarray) which is indexed by another,
    check the indices are sensible.

    TODO: check integers as well?

    Parameters
    ----------
    seq : sequence
    seq_idx : sequence of int

    """

    # Check: minimum index is greater than zero
    if min(seq_idx) < 0:
        raise IndexError('Found index < 0.')

    # Check maximum index is equal to length of sequence - 1
    if max(seq_idx) > len(seq) - 1:
        raise IndexError('Found index larger than sequence length.')
