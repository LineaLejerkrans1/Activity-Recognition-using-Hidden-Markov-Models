import numpy as np
from .DiscreteD import DiscreteD

class MarkovChain:
    """
    MarkovChain - class for first-order discrete Markov chain,
    representing discrete random sequence of integer "state" numbers.
    
    A Markov state sequence S(t), t=1..T
    is determined by fixed initial probabilities P[S(1)=j], and
    fixed transition probabilities P[S(t) | S(t-1)]
    
    A Markov chain with FINITE duration has a special END state,
    coded as nStates+1.
    The sequence generation stops at S(T), if S(T+1)=(nStates+1)
    """
    def __init__(self, initial_prob, transition_prob):

        self.q = initial_prob  #InitialProb(i)= P[S(1) = i]
        self.A = transition_prob #TransitionProb(i,j)= P[S(t)=j | S(t-1)=i]


        self.nStates = transition_prob.shape[0]

        self.is_finite = False
        if self.A.shape[0] != self.A.shape[1]:
            self.is_finite = True


    def probDuration(self, tmax):
        """
        Probability mass of durations t=1...tMax, for a Markov Chain.
        Meaningful result only for finite-duration Markov Chain,
        as pD(:)== 0 for infinite-duration Markov Chain.
        
        Ref: Arne Leijon (201x) Pattern Recognition, KTH-SIP, Problem 4.8.
        """
        pD = np.zeros(tmax)

        if self.is_finite:
            pSt = (np.eye(self.nStates)-self.A.T)@self.q

            for t in range(tmax):
                pD[t] = np.sum(pSt)
                pSt = self.A.T@pSt

        return pD

    def probStateDuration(self, tmax):
        """
        Probability mass of state durations P[D=t], for t=1...tMax
        Ref: Arne Leijon (201x) Pattern Recognition, KTH-SIP, Problem 4.7.
        """
        t = np.arange(tmax).reshape(1, -1)
        aii = np.diag(self.A).reshape(-1, 1)
        
        logpD = np.log(aii)*t+ np.log(1-aii)
        pD = np.exp(logpD)

        return pD

    def meanStateDuration(self):
        """
        Expected value of number of time samples spent in each state
        """
        return 1/(1-np.diag(self.A))
    
    def rand(self, tmax):
        """
        S=rand(self, tmax) returns a random state sequence from given MarkovChain object.
        
        Input:
        tmax= scalar defining maximum length of desired state sequence.
           An infinite-duration MarkovChain always generates sequence of length=tmax
           A finite-duration MarkovChain may return shorter sequence,
           if END state was reached before tmax samples.
        
        Result:
        S= integer row vector with random state sequence,
           NOT INCLUDING the END state,
           even if encountered within tmax samples
        If mc has INFINITE duration,
           length(S) == tmax
        If mc has FINITE duration,
           length(S) <= tmaxs
        """
        
        #*** Insert your own code here 
        

        # Empty array for the random seqence of states to be created 
        S = []

        # Draw inital probobiliteies
        d_init = DiscreteD(self.q)
        # Sample the initial state
        current_state = int(d_init.rand(1)[0, 0])
        S.append(current_state)
        
        # Loop through every time step
        for t in range(1, tmax):
            # Get probobilities from the transition matrix 
            p_transition = self.A[current_state - 1, :]
            d_trans = DiscreteD(p_transition)
            # Sample the next state
            next_state = int(d_trans.rand(1)[0, 0])
            
            # Checks if the loop should be terminated if the states reached is an "end" state
            if self.is_finite and next_state == self.nStates + 1:
                break
                
            # Add the state to the list and move to the next 
            S.append(next_state)
            current_state = next_state
            
        return np.array(S).reshape(1, -1)
        


    def viterbi(self, pX):
        """
        Calculates the most probable state sequence using the Viterbi algorithm.
        Works in log-space to prevent numerical underflow.
        """
        N, T = pX.shape
        
        # Use log probabilities, adding small epsilon to avoid log(0)
        eps = 1e-100
        log_q = np.log(np.maximum(self.q, eps))
        log_A = np.log(np.maximum(self.A[:N, :N], eps))
        log_pX = np.log(np.maximum(pX, eps))
        
        delta = np.zeros((N, T))
        psi = np.zeros((N, T), dtype=int)
        
        # Initialization
        delta[:, 0] = log_q + log_pX[:, 0]
        
        # Recursion
        for t in range(1, T):
            for j in range(N):
                # Calculate probabilities of transitioning from any state i to state j
                trans_probs = delta[:, t-1] + log_A[:, j]
                # Store the max probability
                delta[j, t] = np.max(trans_probs) + log_pX[j, t]
                # Store the argmax 
                psi[j, t] = np.argmax(trans_probs)
                
        # Termination
        opt_states = np.zeros(T, dtype=int)
        opt_states[T-1] = np.argmax(delta[:, T-1])
        max_prob = np.max(delta[:, T-1])
        
        # Backtracking
        for t in range(T-2, -1, -1):
            opt_states[t] = psi[opt_states[t+1], t+1]
            
        return opt_states, max_prob
    
    def stationaryProb(self):
        pass
    
    def stateEntropyRate(self):
        pass
    
    def setStationary(self):
        pass

    def logprob(self):
        pass

    def join(self):
        pass

    def initLeftRight(self):
        pass
    
    def initErgodic(self):
        pass

    def forward(self, pX):
        """
        Calculates the scaled forward variables alpha_hat and scaling factors c.
        Input:
        pX: Matrix of size (N, T) containing emission probabilities b_j(x_t).
        """
        N, T = pX.shape
        # stores the scaled forward variables for each state at each time t
        alpha_hat = np.zeros((N, T))
        # scaleing factor 
        c = np.zeros(T)
        
        # Initialization 
        # alpha = pi_i * b_i(x_1)
        alpha_temp = self.q * pX[:, 0] 
        # calc scaling
        c[0] = 1.0 / np.sum(alpha_temp) 
        # normilize to avoid underflow
        alpha_hat[:, 0] = c[0] * alpha_temp
        
        # Loop over time steps
        for t in range(1, T):
            # next state probabilities using the transition matrix A
            next_state_prob = alpha_hat[:, t-1] @ self.A[:, :N]
            # multiply by emission probability b_j(x_t) for the current observation
            alpha_temp = next_state_prob * pX[:, t]

            # normilize
            c[t] = 1.0 / np.sum(alpha_temp)
            alpha_hat[:, t] = c[t] * alpha_temp
            
        # Finite check
        if self.is_finite:
            # account for the transition to the end state
            a_end = self.A[:, N]
            alpha_end = np.sum(alpha_hat[:, T-1] * a_end)
            c_end = 1.0 / alpha_end
            c = np.append(c, c_end)
            
        return alpha_hat, c


    def finiteDuration(self):
        pass
    
    def backward(self, c, pX):
        """
        Calculates the scaled backward variables beta_hat.
        Input:
        c: Vector of scaling factors (length T for infinite, T+1 for finite)
        pX: Matrix of size (N, T) containing emission probabilities b_j(x_t).
        """
        N, T = pX.shape
        # conditional probabilities of future observations
        beta_hat = np.zeros((N, T))
        
        # Initialization, starts at the end
        if self.is_finite:
            # if finite use the exit probabilities
            beta_hat[:, T-1] = c[T-1] * c[T] * self.A[:, N]
        else:
            # otherwise just ones
            beta_hat[:, T-1] = c[T-1] * np.ones(N)


        # Iterate backward 
        for t in range(T-2, -1, -1):
            # probability of future observations
            beta_hat[:, t] = c[t] * (self.A[:, :N] @ (pX[:, t+1] * beta_hat[:, t+1]))
            
        return beta_hat


    def adaptStart(self):
        """Resets accumulators for the transition matrix and initial probabilities,
        for a new traing loop."""
        self.gamma_1_sum = np.zeros(self.nStates)
        self.xi_sum = np.zeros((self.nStates, self.nStates))

    
    def adaptSet(self):
        """Updates q and A based on accumulated statistics using Laplace Smoothing."""
        # Update initial probabilities q
        self.q = self.gamma_1_sum / np.sum(self.gamma_1_sum)
        
        # Adds a smmothing factor to keep transition paths open
        smoothing_factor = 1e-3  

        # Update transition matrix A 
        for i in range(self.nStates):
            # Add smoothing to the accumulated counts
            smoothed_counts = self.xi_sum[i, :self.nStates] + smoothing_factor
            row_sum = np.sum(smoothed_counts)
            
            if row_sum > 0:
                self.A[i, :self.nStates] = smoothed_counts / row_sum


    def adaptAccum(self, gamma, xi):
        """Accumulates probabilities over the sequence."""
        # Accumulate initial state probability
        self.gamma_1_sum += gamma[:, 0]
        
        # Accumulate transitions over time
        self.xi_sum += np.sum(xi, axis=2)
