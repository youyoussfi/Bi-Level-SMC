import sys

sys.path.append("/workspaces/BI-LEVEL-SMC")

import pickle
import time
import pandas as pd

from paper.simulation.simulate_data import simulate_data
from paper.build_smc import bi_level_SMC
from paper.simulation.simulation_config import settings

# Parameters for data simulation
p_group = 5
p_ext = 5
cov_var = 0.5
beta = 0.5

# Parameters for SMC
N = 25000
P = 200
n_runs = 10
n_procs = -1

time_measure = []
for setting in settings:
    print(setting)
    # Parameters
    n, p_ind, approximation_method, pi_ind = setting
    pi_group = pi_ind

    # Simulate data
    data = simulate_data(
        p_ind=p_ind, p_group=p_group, p_ext=p_ext, cov_var=cov_var, n=n, beta=beta
    )

    # Build and run SMC
    start = time.time()
    smc = bi_level_SMC(
        data=data,
        p_group=p_group,
        p_ind=p_ind,
        nprocs=n_procs,
        N=N,
        P=P,
        nruns=n_runs,
        approximation_method=approximation_method,
        pi_ind=pi_ind,
        pi_group=pi_group,
    )
    end = time.time()

    # Measure run time
    time_measure.append(
        {
            "n": n,
            "p_ind": p_ind,
            "likelihood": approximation_method,
            "time": round((end - start) / 60, 2),
            "prior": pi_ind,
        }
    )

# Save the results
file = open(
    "paper/simulation/results/SMC_output_n_{}_p_{}_am_{}_prior_{}.pkl".format(
        n, p_ind, approximation_method, pi_ind
    ),
    "wb",
)
pickle.dump(smc, file)
file.close()

# Save run time
time_measure = pd.DataFrame(time_measure)
time_measure.to_csv(
    "paper/simulation/results/SMC_time_measure.csv", index=False, sep=";"
)
