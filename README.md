# Activity-Recognition-using-Hidden-Markov-Models
Human activity recognition using Hidden Markov Models (HMMs) to classify smartphone accelerometer data into standing, walking, and running using Baum-Welch training and Viterbi decoding.


## Classes
### 1. `HMM.py`
Acts as the central management engine. It supervises the complete time-series execution pipeline without handling 
low-level mathematical updates directly.

### 2. `MarkovChain.py` 
Governs structural state progression over time. This class is entirely blind to sensor telemetry, operating purely 
on discrete state-space transitions.

### 3. `GaussD.py` (The Physics Emission Engine)
Manages spatial clustering geometry across continuous three-dimensional space ($x, y, z$). It possesses no concept 
of time, analyzing frames as isolated structural points.

### 4. `DiscreteD.py` (Discrete Value Sampler)
An implementation for discrete probability distributions. It is leveraged as an internal tool by `MarkovChain` to
draw random scalar steps and initialize state configurations using cumulative mass sampling functions.



## Execution

### Required Libraries:
* numpy 
* PattRecClasses defined above

### Running the Pipeline:
Execute the master script from your terminal to begin model training and sequential evaluation:

'''python main.py'''
