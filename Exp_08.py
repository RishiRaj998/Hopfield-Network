import numpy as np
import itertools

def bin2bip(binary_vector):
    """Convert binary vectors to bipolar vectors."""
    return np.where(binary_vector == 0, -1, 1)

def bip2bin(bipolar_vector):
    """Convert bipolar vectors to binary vectors."""
    return np.where(bipolar_vector == -1, 0, 1)

def compute_weights(mem_vectors, zero_diagonal=True):
    """Compute the weight matrix using bipolar outer product encoding."""
    q, n = mem_vectors.shape
    W = np.zeros((n, n))
    for i in range(q):
        W += np.outer(mem_vectors[i], mem_vectors[i])
    if zero_diagonal:
        np.fill_diagonal(W, 0)  # Zero the diagonal
    return W

def lyapunov_energy(state, weights):
    """Calculate the Lyapunov energy of a state."""
    return -0.5 * np.dot(np.dot(state, weights), state)

def update_state_synchronous(state, weights):
    """Update the state synchronously."""
    activation = np.dot(state, weights)
    return np.where(activation >= 0, 1, -1)

def update_state_asynchronous(state, weights):
    """Update the state asynchronously."""
    n = state.shape[0]
    permindex = np.random.permutation(n)
    new_state = state.copy()
    for j in permindex:
        activation = np.dot(state, weights[j])
        new_state[j] = 1 if activation >= 0 else -1
    return new_state

def hopfield_network(mem_vectors, probe, mode='synchronous', max_iter=50, zero_diagonal=True):
    bip_mem_vecs = bin2bip(mem_vectors)
    weights = compute_weights(bip_mem_vecs, zero_diagonal=zero_diagonal)

    signal_vector = bin2bip(probe)
    for _ in range(max_iter):
        if mode == 'synchronous':
            new_state = update_state_synchronous(signal_vector, weights)
        elif mode == 'asynchronous':
            new_state = update_state_asynchronous(signal_vector, weights)
        else:
            raise ValueError("Mode must be 'synchronous' or 'asynchronous'.")

        if np.array_equal(signal_vector, new_state):
            break
        signal_vector = new_state

    recalled_vector = bip2bin(signal_vector)
    return recalled_vector, lyapunov_energy(signal_vector, weights)

# Fundamental memory patterns
mem_vectors = np.array([
    [1, 1, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 1, 1]
])

# Calculate Lyapunov energies for each memory vector
energies = [lyapunov_energy(bin2bip(mem), compute_weights(bin2bip(mem_vectors))) for mem in mem_vectors]
print("Lyapunov Energies:", energies)

# Generate all possible binary vectors of dimension 5
all_binary_vectors = np.array(list(itertools.product([0, 1], repeat=5)))

# Determine stable states for synchronous and asynchronous updates
synchronous_states = {}
asynchronous_states = {}

for binary_vector in all_binary_vectors:
    recalled_sync, _ = hopfield_network(mem_vectors, binary_vector, mode='synchronous')
    recalled_async, _ = hopfield_network(mem_vectors, binary_vector, mode='asynchronous')
    synchronous_states[tuple(binary_vector)] = recalled_sync
    asynchronous_states[tuple(binary_vector)] = recalled_async

# Print stable states for synchronous updates
print("\nStable States for Synchronous Updates:")
for binary_vector, stable_state in synchronous_states.items():
    print(f"Input: {binary_vector} -> Stable State: {stable_state}")

# Print stable states for asynchronous updates
print("\nStable States for Asynchronous Updates:")
for binary_vector, stable_state in asynchronous_states.items():
    print(f"Input: {binary_vector} -> Stable State: {stable_state}")

# Analyze recall behavior with non-zero diagonal elements
synchronous_states_nonzero_diag = {}
asynchronous_states_nonzero_diag = {}

for binary_vector in all_binary_vectors:
    recalled_sync, _ = hopfield_network(mem_vectors, binary_vector, mode='synchronous', zero_diagonal=False)
    recalled_async, _ = hopfield_network(mem_vectors, binary_vector, mode='asynchronous', zero_diagonal=False)
    synchronous_states_nonzero_diag[tuple(binary_vector)] = recalled_sync
    asynchronous_states_nonzero_diag[tuple(binary_vector)] = recalled_async

# Print stable states for synchronous updates with non-zero diagonal
print("\nStable States for Synchronous Updates (Non-Zero Diagonal):")
for binary_vector, stable_state in synchronous_states_nonzero_diag.items():
    print(f"Input: {binary_vector} -> Stable State: {stable_state}")

# Print stable states for asynchronous updates with non-zero diagonal
print("\nStable States for Asynchronous Updates (Non-Zero Diagonal):")
for binary_vector, stable_state in asynchronous_states_nonzero_diag.items():
    print(f"Input: {binary_vector} -> Stable State: {stable_state}")
