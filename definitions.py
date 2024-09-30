"""
definitions.py: file to store the functions to define networks and matrices structures

CONTAINS: - up_binary: function to create a binary uptake matrix
          - met_dir:   function to sample D coefficients from Dirichelet distribution, with fixed sparcity

"""

import numpy as np

#-------------------------------------------------------------------------------------------------------------
# up_binary: function to create a binary uptake matrix

def up_binary(n_s,n_r,n_pref):

    """
    n_s:    int, number of species
    n_r:    int, number of resources
    n_pref: int, number of resources consumed by each species

    RETURNS up_mat: matrix, n_sxn_r, containing uptake preferences (binary matrix)
    """

    up_mat = np.zeros((n_s,n_r))
    # each species has a given number of preferred resources
    for i in range(n_s):
        ones_indices = np.random.choice(n_r, n_pref, replace=False)
        up_mat[i, ones_indices] = 1
    # check that someone eats primary source
    if (up_mat[:,0] == 0).all():
        up_mat[np.random.randint(0, n_s-1),0] = 1

    return up_mat

#-------------------------------------------------------------------------------------------------------------
# met_dir: function to create a D-sampled D matrix with given sparcity

def met_dir(n_r,sparcity):

    met_mat = np.ones((n_r,n_r))*(np.random.rand(n_r, n_r) > sparcity)      # make metabolic matrix sparce
    met_mat[0,:] = 0                                                        # carbon source is not produced
    np.fill_diagonal(met_mat, 0)                                            # diagonal elements should be 0
    # check that at least one thing is produced from primary carbon source
    if (met_mat[:,0] == 0).all():
        met_mat[np.random.randint(0, n_r-1),0] = 1
    # sample all from D. distribution
    for column in range(n_r):
        # Find indices where matrix[:, col_idx] == 1 (non-zero entries)
        non_zero_indices = np.where(met_mat[:, column] == 1)[0]  
        if len(non_zero_indices) > 0:
            # Sample from Dirichlet distribution for non-zero entries
            dirichlet_values = np.random.dirichlet(np.ones(len(non_zero_indices)))
            met_mat[non_zero_indices, column] = dirichlet_values

    return met_mat

