import numpy as np
from numpy.typing import NDArray
from hmmlearn import hmm


def hmm_generator(
    seq_length: int,
    states: list[str],
    bases: list[str],
    transition: NDArray[np.float64],
    emission: NDArray[np.float64],
    pi: NDArray[np.float64],
) -> tuple[list, str]:
    """
    Generate sequence from given hidden states matrix and emission matrix.

    Args:
        - seq_length (int): The length of sequence to generate
        - states (list[str]): The list of states
        - bases (list[str]): The list of oberservations
        - transition (NDArray): Transition probability matrix, shape (N, N)
        - emission (NDArray): Emission probability matrix, shape (N, M)
        - pi (NDArray): Initial state probability distribution, shape (N,)

    Return:
        - hidden_path (list[str]): The list of hidden states
        - observed_sequence (str): Observation sequence
    """

    hidden_path = []
    observed_sequence = ""

    map_states = {state: idx for idx, state in enumerate(states)}

    # Initialize hidden state
    current_hidden = np.random.choice(states, p=pi)

    for t in range(seq_length):
        hidden_path.append(current_hidden)
        cur_hidden_idx = map_states.get(current_hidden)

        # Get emission prob according to current hidden state
        emission_prob = emission[cur_hidden_idx]
        # Get current emission observation
        if emission_prob.sum() > 1e-3:
            cur_base = np.random.choice(bases, p=emission_prob)
            observed_sequence += cur_base

        # Get next hidden states according to current hidden states
        transition_prob = transition[cur_hidden_idx]
        current_hidden = np.random.choice(states, p=transition_prob)

    return (hidden_path, observed_sequence)


def log_viterbi(
    obs: NDArray,
    state_size: int,
    pi: NDArray[np.float64],
    trans: NDArray[np.float64],
    emis: NDArray[np.float64],
) -> tuple[NDArray, NDArray, NDArray]:
    """
    Computes the most likely sequence of hidden states using log-space Viterbi.

    Args:
        - obs (NDArray): A list of observation indices, shaps (T,)
        - states (int): The number of states (N)
        - pi (NDArray): Initial state probability distribution, shape (N,)
        - trans (NDArray): Transition probability matrix, shape (N, N)
        - emis (NDArray): Emission probability matrix, shape (N, M)

    Return:
        - best_path (NDArray): The prediction hidden states
        - delta (NDArray): A (N, T) Viterbi Matrix to store the maximum probabilites given observation o_t is from hidden states i
        - psi (NDArray): A (N, T) backpointer tracking matrix to store which previous states had the maximum probability
    """
    T = len(obs)
    N = state_size

    # 1. Convert all probabilites to log-space (use small epsilon to avoid log(0) error)
    epsilon = 1e-100
    log_pi = np.log(pi + epsilon)
    log_trans = np.log(trans + epsilon)
    log_emis = np.log(emis + epsilon)

    # 2. Create Viterbi matrix (delta) and Backpointer tracking matrix (psi)
    delta = np.zeros(
        (N, T), dtype=float
    )  # The maximum probabilites that the given obervation o_t is from hidden state i, i=1, ..., N
    psi = np.zeros(
        (N, T), dtype=int
    )  # Log which previous states had the maximum probability.

    # 3. Computing best path
    # 3-1 Initialization
    delta[:, 0] = log_pi + log_emis[:, obs[0]]

    # 3-2 Recursion
    for t in range(1, T):
        for j in range(N):
            # max_i [ delta_i(t-1) + log_trans(i -> j) ] + log_emit(j -> obs_t)
            trans_scores = delta[:, t - 1] + log_trans[:, j]
            psi[j, t] = np.argmax(trans_scores)
            delta[j, t] = np.max(trans_scores) + log_emis[j, obs[t]]

    # 4. Traceback
    best_path = np.zeros(T, dtype=int)

    best_path[-1] = np.argmax(delta[:, -1])
    for t in range(T - 2, -1, -1):
        best_path[t] = psi[best_path[t + 1], t + 1]

    return best_path, delta, psi


def viterbi_hmmlearn(
    obs: NDArray, state_size: int, pi: NDArray, trans: NDArray, emis: NDArray
) -> NDArray:
    """
    Perform hmm.CategoricalHMM prediction model from hmm learn library.

    Args:
        - obs (NDArray): A list of observation indices, shaps (T,)
        - states (int): The number of states (N)
        - pi (NDArray): Initial state probability distribution, shape (N,)
        - trans (NDArray): Transition probability matrix, shape (N, N)
        - emis (NDArray): Emission probability matrix, shape (N, M)

    Return:
        - best_path (NDArray): The prediction hidden states
    """
    model = hmm.CategoricalHMM(n_components=state_size)

    model.startprob_ = pi
    model.transmat_ = trans
    model.emissionprob_ = emis
    obs = obs.reshape(-1, 1)

    hidden_states = model.predict(obs)
    # log_prob, hidden_states_decode = model.decode(obs)

    return hidden_states
