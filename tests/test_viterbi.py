import unittest

import numpy as np

from hiddenmarkov import HMM, ConstantTransitionModel, CategoricalStringObservationModel


class TestViterbi(unittest.TestCase):

    def test_wikipedia(self):
        obs = ("normal", "cold", "dizzy")
        states = ("Healthy", "Fever")
        observation_probabilities = np.array([[0.5, 0.1],
                                              [0.4, 0.3],
                                              [0.1, 0.6]])
        transition_probabilities = np.array([[0.7, 0.3],
                                             [0.4, 0.6]])

        observation_model = CategoricalStringObservationModel(
            observation_probabilities,
            obs
        )

        init_distribution = np.array([0.6, 0.4])

        transition_model = ConstantTransitionModel(
            transition_probabilities,
            init_distribution)

        hmm = HMM(observation_model, transition_model, state_space=states)

        path, prob = hmm.find_best_sequence(obs, log_probabilities=False)

        self.assertTrue(all(path == ["Healthy", "Healthy", "Fever"]))
        self.assertTrue(np.isclose(prob, 0.01512, atol=1e-5))

        print("Example Wikipedia")
        print("Best sequence", path)
        print("Sequence probability", prob)

    def test_upenn(self):
        # Example taken from https://www.cis.upenn.edu/~cis262/notes/Example-Viterbi-DNA.pdf
        obs = ("A", "C", "G", "T")
        states = ("H", "L")
        observation_probabilities = np.array([[0.2, 0.3],
                                              [0.3, 0.2],
                                              [0.3, 0.2],
                                              [0.2, 0.3]])
        transition_probabilities = np.array([[0.5, 0.5],
                                             [0.4, 0.6]])

        observation_model = CategoricalStringObservationModel(
            observation_probabilities,
            obs
        )

        init_distribution = np.array([0.5, 0.5])

        transition_model = ConstantTransitionModel(
            transition_probabilities,
            init_distribution)

        hmm = HMM(observation_model, transition_model, state_space=states)

        observation = ["G", "G", "C", "A", "C", "T", "G", "A", "A"]
        path, prob = hmm.find_best_sequence(observation, log_probabilities=False)

        self.assertTrue(all(path == ['H', 'H', 'H', 'L', 'L', 'L', 'L', 'L', 'L']))
        self.assertTrue(np.isclose(prob, 4.25e-8))

        print("Example Viterbi-DNA UPenn")
        print("Best sequence", path)
        print("Sequence probability", prob)
