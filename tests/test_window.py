import unittest

import numpy as np

from hiddenmarkov import (HMM, 
                          ConstantTransitionModel, 
                          WindowedObservationModel,
                          create_prob_models)


class TestWindow(unittest.TestCase):
    def test_window(self):
        prob_models =  create_prob_models(no_global_states = 100, 
                                    window_size = 2)
        observation_model = WindowedObservationModel(
            prob_models,
            use_inverted_probs=False,
            use_log_probabilities=True
            )

        transition_probabilities = np.array([[0.7, 0.3],
                                            [0.001, 0.999]])

        init_distribution = np.array([0.999, 0.001])

        transition_model = ConstantTransitionModel(
            transition_probabilities,
            init_distribution=init_distribution,
            use_log_probabilities=True
            )

        hmm = HMM(observation_model, transition_model)

        obs = np.array([ 0,  1,  1,  1,  2,  3,  3,  4,  5,  6,  6,  7,  7,  8,  8,  8,  9,
                        10, 10, 10, 11, 11, 12, 12, 13, 14, 15, 15, 16, 16, 17, 17, 18, 18,
                        19, 19, 19, 19, 20, 20, 20, 21, 22, 22, 23, 24, 25, 25, 26, 26, 27,
                        27, 28, 28, 28, 29, 30, 31, 31, 31, 31, 32, 33, 33, 34, 35, 35, 35,
                        36, 36, 36, 37, 38, 38, 39, 39, 39, 39, 39, 40, 40, 40, 41, 41, 41,
                        41, 41, 41, 41, 42, 42, 42, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
                        52, 52, 53, 54, 54, 55, 55, 55, 55, 56, 57, 58, 58, 59, 60, 60, 60,
                        61, 62, 63, 64, 64, 64, 65, 65, 65, 66, 66, 67, 68, 69, 69, 69, 69,
                        70, 71, 72, 73, 73, 74, 74, 74, 74, 75, 75, 75, 75, 75, 75, 75, 76,
                        77, 78, 78, 78, 78, 79, 79, 80, 81, 81, 81, 82, 83, 83, 83, 83, 83,
                        83, 84, 85, 85, 86, 87, 88, 88, 88, 89, 90, 91, 91, 92, 92, 92, 93,
                        93, 94, 95, 96, 96, 96, 97, 97, 98, 98, 98, 99, 99])

        path, prob = hmm.find_best_sequence(obs, 
                                            log_probabilities=True,
                                            viterbi="windowed")

        self.assertTrue(np.all(path == obs))
        self.assertTrue(np.isclose(prob, -217.51414002612017, atol=1e-5))