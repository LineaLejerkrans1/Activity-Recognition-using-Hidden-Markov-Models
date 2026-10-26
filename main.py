import numpy as np
from PattRecClasses.GaussD import GaussD
from PattRecClasses.MarkovChain import MarkovChain
from PattRecClasses.HMM import HMM


def simple_moving_average(data, window_size=2):
    """
    Applies a moving average filter to smooth out high-frequency noise.
    """
    smoothed = np.zeros_like(data)
    cols = data.shape[1]

    # Loop through X, Y, Z axes
    for i in range(cols):
        weights = np.ones(window_size) / window_size
        # Slides over the data points to create a smoothen value
        smoothed[:, i] = np.convolve(data[:, i], weights, mode='same')
    return smoothed


def load_and_preprocess_file(filename, window_size=2):
    """
    Loads a CSV file, extracts X,Y,Z values, applies a moving average filter.
    """
    raw_matrix = np.loadtxt(filename, delimiter=';', skiprows=1)
    # Use Columns 1, 2 and 3
    X_raw = raw_matrix[:, 1:4] 
    
    # Filter out noise and spikes
    X_filtered = simple_moving_average(X_raw, window_size=window_size)
    return X_filtered.T





# Main function
if __name__ == "__main__":
    print("Loading training files")
    
    # Paths to the training files
    train_paths = {
        "standing": [
            '/Users/linea/Kth/År4/Pattern Recognition/Project/Data/standing.csv'
        ],
        "walking": [
            '/Users/linea/Kth/År4/Pattern Recognition/Project/Data/walking1.csv',
            '/Users/linea/Kth/År4/Pattern Recognition/Project/Data/walking2.csv'
        ],
        "running": [
            '/Users/linea/Kth/År4/Pattern Recognition/Project/Data/running1.csv',
            '/Users/linea/Kth/År4/Pattern Recognition/Project/Data/running2.csv'
        ]
    }
    
    # Combined them into a list for the EM update 
    training_sequences = []
    window_size = 2
    
    for activity, paths in train_paths.items():
        for path in paths:
            seq = load_and_preprocess_file(path, window_size=window_size)
            training_sequences.append(seq)
            print(f"-> Preprocessed {activity} sequence with shape: {seq.shape}")

    print(f"Total training sequences prepared for Baum-Welch EM: {len(training_sequences)}\n")
    

    # Create a HMM
    print("Initializing HMM Model: ")

    # Guess of the matrixes
    q_init = np.array([1/3, 1/3, 1/3]) 
    
    # Start with strong self transition
    A_init = np.array([[0.98, 0.01, 0.01],
                       [0.01, 0.98, 0.01],
                       [0.01, 0.01, 0.98]])

    mc = MarkovChain(initial_prob=q_init, transition_prob=A_init)

    # Place the clusters in the region we expect them to be to help the EM algrigthm 
    # Poor initlilzation could otherwise make the EM gt stuck
    state_distributions = [
        GaussD(means=[0.0, -1.0, 0.0], cov=np.eye(3)*0.15),
        GaussD(means=[0.0, -0.8, 0.0], cov=np.eye(3)*0.15),
        GaussD(means=[0.0, -1.5, 0.0], cov=np.eye(3)*0.15),
    ]

    hmm_model = HMM(mc=mc, distributions=state_distributions)


    print("\nBaum-Welch training:")
    
    # Train the HMM model with the EMM update with 10 iterations
    hmm_model.train(training_sequences, n_iterations=10)
    
    print("\nEM Parameters:")
    print("Transition Matrix (A):")
    print(hmm_model.stateGen.A)
    
    print("\nState Mean Vectors:")
    for i, dist in enumerate(hmm_model.outputDistr):
        print(f"State {i} mean: {dist.means}")
        print(f"State {i} cov diag: {np.diag(dist.cov)}")


    # Evaluate the model:
    test_file_path = '/Users/linea/Kth/År4/Pattern Recognition/Project/Data/test_data.csv'
    X_test = load_and_preprocess_file(test_file_path, window_size=window_size)
    print(f"\n-> Loaded Test sequence, shape: {X_test.shape}")
    
    print("Running Viterbi:")
    predicted_states = hmm_model.viterbi(X_test)

    # Define clusters
    activity_mapping = {
        2: "Running",
        1: "Walking",
        0: "Standing Still"
    }
    
    print("\nTimeline Results of test seqence")
    # Recorded woth 5Hz
    step_interval = 10 
    

    for t in range(0, X_test.shape[1], step_interval):
        state_idx = predicted_states[t]
        activity_name = activity_mapping.get(state_idx, f"Unknown State ({state_idx})")
        print(f"Time Step {t:02d} (~{t*0.2:.1f}s): Predicted Activity -> {activity_name}")

    # Log Likelihood metrics check
    test_log_prob = hmm_model.logprob(X_test)
    print(f"\nTotal Log-Probability of the test sequence: {test_log_prob:.4f}")