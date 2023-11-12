import numpy as np
from scipy.stats import norm

def create_covariates(p: int, n: int, step: int):
    """
    Generate synthetic covariates.

    Parameters:
    - p (int): Number of variables.
    - n (int): Number of observations.
    - step (int): Index of the variable set to 1.

    Returns:
    - Tuple: X (covariates), coeff (coefficients).
    """
    covariance = np.repeat(0.5, p * p).reshape((p, p))
    covariance[np.eye(covariance.shape[0], dtype=bool)] = 1
    X = np.random.multivariate_normal(mean=np.repeat(0, p), cov=covariance, size=n)
    coeff = np.zeros(p)
    if step is not None:
        coeff[step] = 1
    return X, coeff

def create_dictionary(p_ind: int, p_group: int):
    """
    Create a dictionary mapping individual variables to groups.

    Parameters:
    - p_ind (int): Number of individual variables.
    - p_group (int): Number of groups.

    Returns:
    - Dict: Mapping of individual variables to groups.
    """
    step = p_ind // p_group
    return {ind: group for group in range(p_group) for ind in range(group * step, group * step + step)}

def simulate_data(p_ind: int, p_group: int, p_ext: int, n: int):
    """
    Create a synthetic dataset.

    Parameters:
    - p_ind (int): Number of individual variables.
    - p_group (int): Number of group variables.
    - p_ext (int): Number of external variables.
    - n (int): Number of observations.
    - step (int): Index of the variable set to 1.

    Returns:
    - Tuple: X (covariates), y (outcome), dict_group (mapping), p_ext (number of external variables).
    """
    step = int(p_ind/p_group)

    # Generate individual covariates and coefficients
    X_ind, coeff_ind = create_covariates(p=p_ind, n=n, step=np.array([p_ind - (step * 2 + 1), p_ind - (step + 1), p_ind - 1]))

    # Generate group covariates and coefficients
    X_group, coeff_group = create_covariates(p=p_group, n=n, step=np.array([2, 3, 4]))

    # Generate external covariates and coefficients
    X_ext, coeff_ext = create_covariates(p=p_ext, n=n, step=np.array([2, 3, 4]))

    # Concatenate the variables (external, group, individual)
    X = np.concatenate([X_ext, X_group, X_ind], axis=1)
    
    # Concatenate the coefficients (external, group, individual)
    coeff = np.concatenate([coeff_ext, coeff_group, coeff_ind], axis=0)

    # Simulate the outcome by multiplying covariates with coefficients and adding noise
    y = np.matmul(X, coeff) + norm.rvs(size=n)
    
    # Convert the continuous outcome to binary (0 or 1) based on a threshold
    y = np.where(y >= 0, 1, 0)

    # Create a dictionary of group-variable correspondences
    dict_group = create_dictionary(p_ind=p_ind, p_group=p_group)

    return X, y, dict_group, p_ext