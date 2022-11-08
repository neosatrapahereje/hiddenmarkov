#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utility functions
"""
import numpy as np


def inverted_softmax(array: np.ndarray) -> np.ndarray:
    """
    array of distances (non-negative) converted to probabilities
    the lowest distance has the highest probability
    """
    return np.exp(-array) / np.sum(np.exp(-array))


def softmax(array: np.ndarray) -> np.ndarray:
    """
    array of estimations converted to probabilities
    the highest prob has the highest probability
    """
    return np.exp(array) / np.sum(np.exp(array))


def create_prob_models(no_global_states: int = 100, window_size: int = 5) -> dict:
    """
    example function to create a
    list of state-specific probability models
    for use with viterbi_algorithm_windowed
    and WindowedObservationModel

    Parameters
    ----------
    no_global_states : int
        number of global states in the HMM
    window_size : int
        number of states visible to the transition model

    Attributes
    ----------
    prob_models_at_state: dict

    """
    models_per_state = list()
    for i in range(no_global_states):
        # for each global state, create list of prob models
        # corresponding to local states in the window
        models = list()
        for j in range(window_size):
            # for each local state, create a prob models
            def probf(input, ref=i + j):
                # dummy prob function
                return int(input == ref)

            models.append((probf, max(0, min(no_global_states - 1, i + j))))
        models_per_state.append(models)
    return models_per_state
