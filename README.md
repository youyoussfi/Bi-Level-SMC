# Scalable Bayesian Bi-level Variable Selection in Generalized Linear Models

**Authors:** Younès Youssfi, Nicolas Chopin

**DOI:** [10.48550/arXiv.2303.12462](https://doi.org/10.48550/arXiv.2303.12462)

## Abstract

Motivated by a real-world application in cardiology, this work presents an algorithm for performing Bayesian bi-level variable selection in generalized linear models. The proposed algorithm is designed to handle large datasets in terms of both the number of individuals and predictors. It builds upon the waste-free SMC (Sequential Monte Carlo) methodology of Dau and Chopin (2022), introduces a novel proposal mechanism to address constraints specific to bi-level selection, and utilizes the ALA (Approximate Laplace Approximation) approach of Rossell et al. (2021).

## Code Organization

The code is organized as follows:

```
.
├── README.md
├── paper
│   └── simulation
│       ├── __init__.py
│       ├── build_smc.py
│       ├── results
│       ├── run_simulation.py
│       ├── simulate_data.py
│       └── simulation_config.py
├── particles
│   ├── __init__.py
│   ├── binary_smc.py
│   ├── collectors.py
│   ├── core.py
│   ├── distributions.py
│   ├── hilbert.py
│   ├── qmc.py
│   ├── resampling.py
│   ├── smc_samplers.py
│   ├── smoothing.py
│   ├── state_space_models.py
│   └── utils.py
└── requirements.txt

```

The `particles` folder is based on the Particles package developed by Nicolas Chopin: [Particles Package](https://github.com/nchopin/particles).

The `paper/simulation` folder contains all the materials needed to reproduce the simulated numerical experiments of the paper:
- `build_smc.py`: builds the SMC algorithm.
- `simulate_data.py`: creates synthetic data to run the SMC algorithm.
- `simulation_config.py`: contains all the configuration settings for the numerical experiment
- `run_simulation.py`: runs the SMC algorithm for the configuration settings given in `simulation_config.py`


## Installation

Use [pip](https://pip.pypa.io/en/stable/) to install the Python libraries specified in the `requirements.txt` file
```
pip install -r requirements.txt
```

## Example 

```python

from paper.simulation.simulate_data import simulate_data
from paper.build_smc import bi_level_SMC  
from paper.simulation.simulation_config import settings

import pickle 
import numpy

# Build and run the SMC for bi-level variable selection

# Parameters :
# - p_group: Number of group variables
# - p_ext: Number of external variables
# - p_ind: Number of individual variables
# - cov_var: Value of the variance-covariance matrix
# - n: Number of observations
# - nprocs: Number of cores used to launch the algorithm 
# - N: Number of particles
# - P: Length of the Markov chains.
# - nruns: Number of runs
# - approximation_method: Likelihood approximation method ('ALA' or 'LA')
# - pi_ind: Parameter of the Bernoulli prior distribution for individual variables
# - pi_group: Parameter of the Bernoulli prior distribution for group variables

p_ind = 50
p_group = 5

# Simulate data
data = simulate_data(p_ind=p_ind, 
                     p_group=p_group, 
                     p_ext=5, 
                     cov_var=0.5,
                     n=100)

# Build and run SMC
smc = bi_level_SMC(data=data,
                   p_group=p_group, 
                   p_ind=p_ind, 
                   nprocs=-5,
                   N=10000,
                   P=1000,
                   nruns=5,
                   approximation_method='ALA',
                   pi_ind=0.5,
                   pi_group=0.5)

# Get posterior probabilities of inclusion
run = 0
post_prob = numpy.mean(smc[run]['output'].X.theta, axis=0)
group_prob = post_prob[:p_group])
ind_prob = post_prob[p_group:])