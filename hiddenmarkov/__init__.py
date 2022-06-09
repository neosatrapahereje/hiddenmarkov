# -*- coding: utf-8 -*-
"""
Hidden Markov Models

This module contains case classes to define Hidden Markov Models.
"""
import pkg_resources
import warnings
import numpy as np
from collections import defaultdict

# define a version variable
__version__ = pkg_resources.get_distribution("hiddenmarkov").version


class TransitionModel(object):
    """
    Base class for implementing a Transition Model
    """
    def __init__(self, use_log_probabilities=True):
        self.use_log_probabilities = use_log_probabilities

    def __call__(self, i=None, j=None, *args, **kwargs):
        raise NotImplementedError


class ObservationModel(object):
    """
    Base class for implementing an Observation Model
    """
    def __init__(self, use_log_probabilities=True):
        self.use_log_probabilities = use_log_probabilities

    def __call__(self, observation, *args, **kwargs):
        raise NotImplementedError()


class HiddenMarkovModel(object):
    """
    Hidden Markov Model

    Parameters
    ----------
    observation_model : ObservationModel
        A model for computiong the observation (emission) probabilities.
    transition_model: TransitionModel
        A model for computing the transition probabilities.
    state_space: iterable (optional)
        Labels of the states (e.g., a list of strings containing
        the names of each state).

    Attributes
    ----------
    observation_model: ObservationModel
    transition_model: TransitionModel
    n_states: int
        Number of states
    state_space: np.array

    TODO
    ----
    * Add forward algorithm
    """

    def __init__(self, observation_model,
                 transition_model, 
                 state_space=None):

        self.observation_model = observation_model
        self.transition_model = transition_model
        self.n_states = self.transition_model.n_states
        if state_space is not None:
            self.state_space = np.asarray(state_space)
        else:
            self.state_space = np.arange(self.n_states)

    def find_best_sequence(self, 
                           observations, 
                           log_probabilities=True,
                           viterbi="optimized"):
        if viterbi == "optimized":
            viterbi_fun = viterbi_algorithm_optimized
        elif viterbi == "windowed":
            viterbi_fun = viterbi_algorithm_optimized_windowed
        elif viterbi == "legacy":
            viterbi_fun = viterbi_algorithm
        else:
            warnings.warn("viterbi needs to be 'optimized', 'windowed', or 'legacy'")
            return
        best_sequence, sequence_likelihood = viterbi_fun(
            hmm=self,
            observations=observations,
            log_probabilities=log_probabilities)
        return best_sequence, sequence_likelihood

# alias
HMM = HiddenMarkovModel


def viterbi_algorithm(hmm, observations, log_probabilities=True):
    """
    Find the most probable sequence of latent variables given
    a sequence of observations

    Parameters
    ----------
    observations: iterable
       An iterable containing observations. The type of each
       element depends on input types accepted by the
       `hmm.observation_model`
    log_probabilities: Bool (optional)
       If True, uses log probabilities to compute the Viterbi
       recursion (better for numerical stability). Default is True.

    Returns
    -------
    path: np.ndarray
        The most probable sequence of latent variables
    likelihood: float
        The likelihood (either the probability or the
        log proability if `log_probabilities` is True)
        of the best sequence.
    """
    # Set whether to use log probabilities in transition and
    # observation models
    hmm.transition_model.use_log_probabilities = log_probabilities
    hmm.observation_model.use_log_probabilities = log_probabilities
    # Initialize matrix for holding the best sub-sequence
    # (log-)likelihood
    omega = np.zeros((len(observations), hmm.n_states))
    # Initialize dictionary for tracking the best paths
    path = defaultdict(lambda: list())

    # Initiate for i == 0
    obs_prob = hmm.observation_model(observations[0])

    if log_probabilities:
        omega[0, :] = obs_prob + hmm.transition_model.init_distribution
    else:
        omega[0, :] = obs_prob * hmm.transition_model.init_distribution

    # Viterbi recursion
    for i, obs in enumerate(observations[1:], 1):
        obs_prob = hmm.observation_model(obs)
        for j in range(hmm.n_states):
            if log_probabilities:
                prob, state = max(
                    [(omega[i - 1, k] + hmm.transition_model(k, j), k)
                     for k in range(hmm.n_states)],
                    key=lambda x: x[0]
                )
                omega[i, j] = obs_prob[j] + prob

            else:
                prob, state = max(
                    [(omega[i - 1, k] * hmm.transition_model(k, j), k)
                     for k in range(hmm.n_states)],
                    key=lambda x: x[0]
                )
                omega[i, j] = obs_prob[j] * prob
            # keep track of the best state
            path[j].append(state)

    # Get best path (backtracking!)
    # Get index of the best state
    best_sequence_idx = omega[-1, :].argmax()
    # likelihood of the path
    path_likelihood = omega[-1, best_sequence_idx]
    # follow the best path backwards
    seq = [best_sequence_idx]
    for s in range(len(path[best_sequence_idx])):
        best_sequence_idx = path[best_sequence_idx][-(s+1)]
        seq.append(best_sequence_idx)
    # invert the path
    best_sequence = np.array(seq[::-1], dtype=int)

    if hmm.state_space is not None:
        best_sequence = hmm.state_space[best_sequence]

    return best_sequence, path_likelihood


def viterbi_algorithm_optimized(hmm, observations, log_probabilities=True):
    """
    Find the most probable sequence of latent variables given
    a sequence of observations

    Parameters
    ----------
    observations: iterable
       An iterable containing observations. The type of each
       element depends on input types accepted by the
       `hmm.observation_model`
    log_probabilities: Bool (optional)
       If True, uses log probabilities to compute the Viterbi
       recursion (better for numerical stability). Default is True.

    Returns
    -------
    path: np.ndarray
        The most probable sequence of latent variables
    likelihood: float
        The likelihood (either the probability or the
        log proability if `log_probabilities` is True)
        of the best sequence.
    """
    # Set whether to use log probabilities in transition and
    # observation models
    hmm.transition_model.use_log_probabilities = log_probabilities
    hmm.observation_model.use_log_probabilities = log_probabilities
    # Initialize matrix for holding the best sub-sequence
    # (log-)likelihood
    omega = np.zeros((len(observations), hmm.n_states))
    # Initialize matrix for holding the best sub-sequence idx
    omega_idx = np.zeros((len(observations), hmm.n_states), dtype=int)

    # Initiate for i == 0
    obs_prob = hmm.observation_model(observations[0])

    if log_probabilities:
        omega[0, :] = obs_prob + hmm.transition_model.init_distribution
    else:
        omega[0, :] = obs_prob * hmm.transition_model.init_distribution

    omega_idx[0, :] = 0

    # Viterbi recursion
    if log_probabilities:
        for i, obs in enumerate(observations[1:], 1):
            obs_prob = hmm.observation_model(obs)
            # omega slice is a row vector, transition_model is a matrix 
            # of prob from state id_row to state id_column
            prob_of_jump_to_state = omega[i - 1, :] + hmm.transition_model().T
            state = np.argmax(prob_of_jump_to_state, axis = 1)
            prob = prob_of_jump_to_state[np.arange(hmm.n_states),state]
            omega[i, :] = obs_prob + prob
            omega_idx[i, :] = state
            
    else:
        for i, obs in enumerate(observations[1:], 1):
            obs_prob = hmm.observation_model(obs)
            # omega slice is a row vector, transition_model is a matrix 
            # of prob from state id_row to state id_column
            prob_of_jump_to_state = omega[i - 1, :] * hmm.transition_model().T
            state = np.argmax(prob_of_jump_to_state, axis = 1)
            prob = prob_of_jump_to_state[np.arange(hmm.n_states),state]
            omega[i, :] = obs_prob * prob
            omega_idx[i, :] = state

    # Get best path (backtracking!)
    # Get index of the best state
    best_sequence_idx = omega[-1, :].argmax()
    # likelihood of the path
    path_likelihood = omega[-1, best_sequence_idx]
    # Get best path (backtracking!)
    seq = [best_sequence_idx]
    for s in range(len(observations) - 1):
        best_sequence_idx = omega_idx[-(s+1), best_sequence_idx]
        seq.append(best_sequence_idx)
    best_sequence = np.array(seq[::-1], dtype=int)
    if hmm.state_space is not None:
        best_sequence = hmm.state_space[best_sequence]

    return best_sequence, path_likelihood


def viterbi_algorithm_optimized_windowed(hmm, 
                                         observations, 
                                         log_probabilities=True):
    """
    Find the most probable sequence of latent variables given
    a sequence of observations.
    
    !!! This version uses a transition model with fixed window of
    states (see WindowedHiddenMarkiovModel). This window represents
    the currently "visible" states from any of an underlying
    longer sequence of global states.
    
    The viterbi uses the current best path at each step to 
    update the global path. This effectively limits the path
    to corridor in the states. The observation model is called 
    with a global state id, so the true, underlying states
    can have different probabilities.
      

    Parameters
    ----------
    observations: iterable
       An iterable containing observations. The type of each
       element depends on input types accepted by the
       `hmm.observation_model`
    log_probabilities: Bool (optional)
       If True, uses log probabilities to compute the Viterbi
       recursion (better for numerical stability). Default is True.

    Returns
    -------
    path: np.ndarray
        The most probable sequence of latent variables
    likelihood: float
        The likelihood (either the probability or the
        log proability if `log_probabilities` is True)
        of the best sequence.
    """
    # Set whether to use log probabilities in transition and
    # observation models
    hmm.transition_model.use_log_probabilities = log_probabilities
    hmm.observation_model.use_log_probabilities = log_probabilities
    # Initialize matrix for holding the best sub-sequence
    # (log-)likelihood
    omega = np.zeros((len(observations), hmm.n_states))
    # Initialize matrix for holding the best sub-sequence idx
    omega_idx = np.zeros((len(observations), hmm.n_states), dtype=int)
    
    # Initialize vector for holding the current best idx
    # Current window id; start at state 0 and keep the state at the start of the current window
    current_window_idx = np.zeros((len(observations)+1), dtype=int)

    omega_idx[0, :] = 0
    current_window_idx[0] = 0 # zero is a dummy state just for jump size
    current_window_idx[1] = 0 # zero is the actual first window idx

    # Initiate for i == 0
    obs_prob, _ = hmm.observation_model(observations[0], current_window_idx[0])

    if log_probabilities:
        omega[0, :] = obs_prob + hmm.transition_model.init_distribution
    else:
        omega[0, :] = obs_prob * hmm.transition_model.init_distribution

    # Viterbi recursion
    if log_probabilities:
        for i, obs in enumerate(observations[1:], 1):
            obs_prob, glob_ref_idx = hmm.observation_model(obs, current_window_idx[i])
            # omega slice is a row vector, transition_model is a matrix 
            # of prob from state id_row to state id_column
            # use only the slice of omega that is shifted by the previous jump 
            previous_jump = current_window_idx[i] - current_window_idx[i-1]
            prob_of_jump_to_state = np.concatenate((
                                                    omega[i - 1, previous_jump:],
                                                    np.ones(previous_jump)*-np.inf
                                                    )) + \
                                    hmm.transition_model().T
            state = np.argmax(prob_of_jump_to_state, axis = 1)
            prob = prob_of_jump_to_state[np.arange(hmm.n_states),state]
            omega[i, :] = obs_prob + prob
            omega_idx[i, :] = state
            # slide the window to the current best
            current_best_sequence_idx = omega[i, :].argmax()
            current_window_idx[i+1] = glob_ref_idx[current_best_sequence_idx]   
            
    else:
        for i, obs in enumerate(observations[1:], 1):
            obs_prob, glob_ref_idx = hmm.observation_model(obs, current_window_idx[i])
            # omega slice is a row vector, transition_model is a matrix 
            # of prob from state id_row to state id_column
            # use only the slice of omega that is shifted by the previous jump
            previous_jump = current_window_idx[i] - current_window_idx[i-1]
            prob_of_jump_to_state = np.concatenate((
                                                    omega[i - 1, previous_jump:],
                                                    np.zeros(previous_jump)
                                                    )) * \
                                    hmm.transition_model().T
            state = np.argmax(prob_of_jump_to_state, axis = 1)
            prob = prob_of_jump_to_state[np.arange(hmm.n_states),state]
            omega[i, :] = obs_prob * prob
            omega_idx[i, :] = state
            # slide the window to the current best
            current_best_sequence_idx = omega[i, :].argmax()
            current_window_idx[i+1] = glob_ref_idx[current_best_sequence_idx]

    # Get best path (backtracking!)
    # Get index of the best state
    best_sequence_idx = omega[-1, :].argmax()
    # likelihood of the path
    path_likelihood = omega[-1, best_sequence_idx]

    return current_window_idx[1:], path_likelihood


class ConstantTransitionModel(object):
    """
    Constant Transition Model

    This transition model represents the case were the
    transition proabilities do not change over time (i.e.,
    they are static). In this case, the transition probabilities
    can be represented by a transition matrix

    Parameters
    ----------
    transition_probabilities: np.ndarray
        A (n_states, n_states) matrix where component
        [i, j] represents the probability of going to state j
        coming from state i.
    init_distribution: np.ndarray or None (optional)
        A 1D vector of length n_states defining the initial
        probabilities of each state
    normalize_init_distribution: Bool (optional)
        If True, the initial distribution will be normalized.
        Default is False.
    use_log_probabilities: Bool (optional)
        If True, use log proabilities instead of norm proabilities
        (better for numerical stability)
    """

    def __init__(
            self,
            transition_probabilities,
            init_distribution=None,
            normalize_init_distribution=False,
            normalize_transition_probabilities=False,
            use_log_probabilities=True
    ):
        super().__init__()
        self.use_log_probabilities = use_log_probabilities
        self.transition_probabilities = transition_probabilities
        self.n_states = len(transition_probabilities)

        if init_distribution is None:
            self.init_distribution = (
                1.0 / float(self.n_states) *
                np.ones(self.n_states, dtype=float)
            )
        else:
            self.init_distribution = init_distribution

        if normalize_init_distribution:
            # Normalize initial distribution
            self.init_distribution /= np.maximum(
                np.sum(self.init_distribution), 1e-10
            )

        if normalize_transition_probabilities:
            self.transition_probabilities /= np.sum(
                self.transition_probabilities, 1,
                keepdims=True
            )

    @property
    def init_distribution(self):
        if self.use_log_probabilities:
            return self._log_init_dist
        else:
            return self._init_dist

    @init_distribution.setter
    def init_distribution(self, init_distribution):
        self._init_dist = init_distribution
        self._log_init_dist = np.log(self._init_dist)

    @property
    def transition_probabilities(self):
        if self.use_log_probabilities:
            return self._log_transition_prob
        else:
            return self._transition_prob

    @transition_probabilities.setter
    def transition_probabilities(self, transition_probabilities):
        self._transition_prob = transition_probabilities
        self._log_transition_prob = np.log(self._transition_prob)

    def __call__(self, i=None, j=None):
        if i is None and j is None:
            return self.transition_probabilities
        elif i is not None and j is None:
            return self.transition_probabilities[i, :]
        elif i is None and j is not None:
            return self.transition_probabilities[:, j]
        else:
            return self.transition_probabilities[i, j]


class CategoricalStringObservationModel(ObservationModel):

    def __init__(
            self,
            observation_probabilities,
            observations=None,
            use_log_probabilities=True
    ):
        super().__init__(use_log_probabilities=use_log_probabilities)

        self.observation_probabilities = observation_probabilities

        if observations is not None:
            self.observations = list(observations)
        else:
            self.observations = [
                str(i) for i in range(len(observation_probabilities))
            ]

    @property
    def observation_probabilities(self):
        if self.use_log_probabilities:
            return self._log_obs_prob
        else:
            return self._obs_prob

    @observation_probabilities.setter
    def observation_probabilities(self, observation_probabilities):
        self._obs_prob = observation_probabilities
        self._log_obs_prob = np.log(self._obs_prob)

    def __call__(self, observation, *args, **kwargs):
        idx = self.observations.index(observation)
        return self.observation_probabilities[idx]


class WindowedObservationModel(ObservationModel):
    """
    Windowed Observation Model
    Uses a list of state-specific probability models
    as observation model. Probability models can also be distances,
    in which case the different distances are inverted and softmaxed.
    
    Only works with viterbi_algorithm_optimized_windowed
    
    Parameters
    ----------
    prob_models_at_state : list of tuples
        A list of tuples (probability_model,prob_model_state_id) 
        indexed by state_id
    
    Attributes
    ----------
    prob_models_at_state: list of tuples

    """
    def __init__(
            self,
            prob_models_at_state,
            use_inverted_probs=True,
            use_log_probabilities=True
    ):
        super().__init__(use_log_probabilities=use_log_probabilities)
        self.use_log_probabilities = use_log_probabilities
        self.use_inverted_probs = use_inverted_probs
        self.prob_models_at_state = prob_models_at_state
        self.state_number = len(prob_models_at_state)
        if self.use_inverted_probs:
            if self.use_log_probabilities:
                self.outfunc = lambda d : np.log(inverted_softmax(np.array(d)))
            else:
                self.outfunc = lambda d : inverted_softmax(np.array(d))
        else:
            if self.use_log_probabilities:
                self.outfunc = lambda d : np.log(softmax(np.array(d)))
            else:
                self.outfunc = lambda d : softmax(np.array(d))

        
    def __call__(self, observation, current_state, *args, **kwargs):
        # give the current state pick the right states 
        prob_models = self.prob_models_at_state[max(0,min(current_state,self.state_number-1))]
        # compute the probability for all models
        dists = list()
        glob_ref_idx = list()
        
        for prob_model, idx in prob_models: 
            dists.append(prob_model(observation))
            glob_ref_idx.append(idx)
        
        return self.outfunc(dists), np.array(glob_ref_idx)
 
    
def inverted_softmax(array):
    """
    array of distances (non-negative) converted to probabilities
    the lowest distance has the highest probability
    """
    return np.exp(-array)/np.sum(np.exp(-array))


def softmax(array):
    """
    array of estimations converted to probabilities
    the highest prob has the highest probability
    """
    return np.exp(array)/np.sum(np.exp(array))


def create_prob_models(no_global_states = 100, 
                       window_size = 5):
    """
    example function to create a
    list of state-specific probability models
    for use with viterbi_algorithm_optimized_windowed
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
            def probf(input, ref = i+j):
                # dummy prob function
                return int(input == ref)
            models.append((probf, max(0,min(no_global_states-1,i+j))))
        models_per_state.append(models)
    return models_per_state