#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for implementations of the viterbi algorithm
"""

import unittest

import numpy as np

from hiddenmarkov import (
    HMM,
    ConstantTransitionModel,
    CategoricalObservationModel,
    viterbi_algorithm,
    viterbi_algorithm_naive,
)


def wikipedia():
    """
    Example from
    https://en.wikipedia.org/wiki/Viterbi_algorithm#Example
    """

    obs = ("normal", "cold", "dizzy")
    observations = ("normal", "cold", "dizzy")
    states = ("Healthy", "Fever")
    observation_probabilities = np.array([[0.5, 0.1], [0.4, 0.3], [0.1, 0.6]])
    transition_probabilities = np.array([[0.7, 0.3], [0.4, 0.6]])

    observation_model = CategoricalObservationModel(
        observation_probabilities, obs
    )

    init_probabilities = np.array([0.6, 0.4])

    transition_model = ConstantTransitionModel(
        transition_probabilities, init_probabilities
    )

    hmm = HMM(observation_model, transition_model, state_space=states)
    expected_prob = 0.01512
    expected_sequence = np.array(["Healthy", "Healthy", "Fever"])
    return hmm, observations, expected_prob, expected_sequence


class TestViterbi(unittest.TestCase):
    def test_viterbi_prob_wikipedia(self):
        hmm, observations, expected_prob, expected_sequence = wikipedia()
        path, prob = viterbi_algorithm(hmm, observations, log_probabilities=False)
        self.assertTrue(np.all(path == expected_sequence))
        self.assertTrue(np.isclose(prob, expected_prob, atol=1e-5))

    def test_viterbi_log_wikipedia(self):
        hmm, observations, expected_prob, expected_sequence = wikipedia()
        path, prob = viterbi_algorithm(hmm, observations, log_probabilities=True)
        self.assertTrue(np.all(path == expected_sequence))
        self.assertTrue(np.isclose(prob, np.log(expected_prob), atol=1e-5))

    def test_find_best_sequence_prob_wikipedia(self):
        hmm, observations, expected_prob, expected_sequence = wikipedia()

        # optimized log prob
        path, prob = hmm.find_best_sequence(
            observations=observations,
            log_probabilities=True,
            viterbi="optimized",
        )
        self.assertTrue(np.all(path == expected_sequence))
        self.assertTrue(np.isclose(prob, np.log(expected_prob), atol=1e-5))

        # optimized prob
        path, prob = hmm.find_best_sequence(
            observations=observations,
            log_probabilities=False,
            viterbi="optimized",
        )
        self.assertTrue(np.all(path == expected_sequence))
        self.assertTrue(np.isclose(prob, expected_prob, atol=1e-5))

        # naive log prob
        path, prob = hmm.find_best_sequence(
            observations=observations,
            log_probabilities=True,
            viterbi="naive",
        )
        self.assertTrue(np.all(path == expected_sequence))
        self.assertTrue(np.isclose(prob, np.log(expected_prob), atol=1e-5))

        # naive prob
        path, prob = hmm.find_best_sequence(
            observations=observations,
            log_probabilities=False,
            viterbi="naive",
        )
        self.assertTrue(np.all(path == expected_sequence))
        self.assertTrue(np.isclose(prob, expected_prob, atol=1e-5))

        # other (expect an error)

        try:
            path, prob = hmm.find_best_sequence(
                observations=observations,
                log_probabilities=False,
                viterbi="other",
            )
            # fail the test if this the above code does not
            # raise an error
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)

    def test_wikipedia_naive(self):
        """
        https://en.wikipedia.org/wiki/Viterbi_algorithm#Example
        """
        hmm, observations, expected_prob, expected_sequence = wikipedia()
        path, prob = viterbi_algorithm_naive(hmm, observations, log_probabilities=False)
        print("Example Wikipedia")
        print("Best sequence", path)
        print("Expected Sequence", ["Healthy", "Healthy", "Fever"])
        print("Sequence probability", prob)

        self.assertTrue(all(path == ["Healthy", "Healthy", "Fever"]))
        self.assertTrue(np.isclose(prob, 0.01512, atol=1e-5))

    def test_upenn(self):
        """
        Example taken from
        https://www.cis.upenn.edu/~cis262/notes/Example-Viterbi-DNA.pdf
        """

        obs = ("A", "C", "G", "T")
        states = ("H", "L")
        observation_probabilities = np.array(
            [[0.2, 0.3], [0.3, 0.2], [0.3, 0.2], [0.2, 0.3]]
        )
        transition_probabilities = np.array([[0.5, 0.5], [0.4, 0.6]])

        observation_model = CategoricalObservationModel(
            observation_probabilities, obs
        )

        init_probabilities = np.array([0.5, 0.5])

        transition_model = ConstantTransitionModel(
            transition_probabilities, init_probabilities
        )

        hmm = HMM(observation_model, transition_model, state_space=states)

        observations = ["G", "G", "C", "A", "C", "T", "G", "A", "A"]
        path, prob = viterbi_algorithm(hmm, observations, log_probabilities=False)
        print("Example Viterbi-DNA UPenn")
        print("Best sequence", path)
        print("Expected sequence", ["H", "H", "H", "L", "L", "L", "L", "L", "L"])
        print("Sequence probability", prob)

        self.assertTrue(all(path == ["H", "H", "H", "L", "L", "L", "L", "L", "L"]))
        self.assertTrue(np.isclose(prob, 4.25e-8))
