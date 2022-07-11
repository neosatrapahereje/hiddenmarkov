# hiddenmarkov

A simple Python library for Hidden Markov Models.

This library was created mostly for educational purposes and does not aim to have the most efficient implementation.

## Setup

Simple installation

```
pip install python-hiddenmarkov
```

Develop version
```bash
git clone https://github.com/neosatrapahereje/hiddenmarkov.git
cd hiddenmarkov
pip install -e .
```

## Usage

Example from [Wikipedia](https://en.wikipedia.org/wiki/Viterbi_algorithm#Example)

```python
obs = ("normal", "cold", "dizzy")
observations = ("normal", "cold", "dizzy")
states = ("Healthy", "Fever")
observation_probabilities = np.array([[0.5, 0.1], [0.4, 0.3], [0.1, 0.6]])
transition_probabilities = np.array([[0.7, 0.3], [0.4, 0.6]])

observation_model = CategoricalStringObservationModel(
	observation_probabilities, obs
)

init_distribution = np.array([0.6, 0.4])

transition_model = ConstantTransitionModel(
	transition_probabilities, init_distribution
)

hmm = HMM(observation_model, transition_model, state_space=states)

path, prob = viterbi_algorithm(hmm, observations, log_probabilities=False)
print("Example Wikipedia")
print("Best sequence", path)
print("Expected Sequence", ["Healthy", "Healthy", "Fever"])
print("Sequence probability", prob)
```

## Licence

The code in this package is licensed under the MIT Licence. For details,
please see the [LICENSE](https://github.com/neosatrapahereje/hiddenmarkov/blob/main/LICENSE) file.
