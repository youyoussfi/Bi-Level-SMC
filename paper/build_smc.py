import build_particles
from build_particles import distributions
from build_particles import smc_samplers
from build_particles import binary_smc


def bi_level_SMC(
    data: tuple,
    p_group: int,
    p_ind: int,
    nprocs: int,
    N: int,
    P: int,
    nruns: int,
    approximation_method: str,
    pi_ind: float,
    pi_group: float,
):
    """
    Build a Sequential Monte Carlo (SMC) algorithm for a bi-level variable selection model.

    Parameters:
    ** Input data: **
    - data (tuple) : Data used to feed the SMC algorithm
    - p_group (int): Number of group variables.
    - p_ind (int): Number of individual variables.

    ** SMC algorithm: **
    - nprocs (int): Number of cores used to run the algorithm.
    - N (int): Number of particles.
    - P (int): Length of the Markov chains.
    - nruns (int): Number of runs.
    - approximation_method (str): Likelihood approximation method ('ALA' or 'LA').
    - pi_ind (float): Parameter of the Bernoulli prior distribution for individual variables.
    - pi_group (float): Parameter of the Bernoulli prior distribution for group variables.

    Returns:
    - particles.SMC: SMC algorithm result.
    """
    # Check if the approximation method is valid
    if approximation_method not in ["ALA", "LA"]:
        raise ValueError(
            "Invalid approximation_method. Supported values are 'ALA' or 'LA'."
        )

    # Step 1: Specify the prior distribution
    prior = distributions.BiLevelPrior(
        p_group=p_group,
        p_ind=p_ind,
        pi_group=pi_group,
        pi_ind=pi_ind,
        dict_group=data[2],
    )

    # Step 3: Specify the likelihood approximation method
    if approximation_method == "ALA":
        model = binary_smc.BilevelALA(data=data, prior=prior)
    elif approximation_method == "LA":
        model = binary_smc.BilevelLA(data=data, prior=prior)

    # Step 4: Specify the MCMC algorithm
    mcmc = smc_samplers.MCMCSequenceWF(
        mcmc=binary_smc.BiLevelBinaryMetropolis(data=data), len_chain=P
    )

    # Step 5: Specify the SMC algorithm
    smc = smc_samplers.AdaptiveTempering(model=model, move=mcmc, len_chain=P)

    # Step 6: Run the SMC algorithm
    selection = build_particles.multiSMC(
        fk=smc, N=N // P, nruns=nruns, nprocs=nprocs, verbose=0
    )

    return selection
