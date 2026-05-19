from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse.csgraph import breadth_first_order
from scipy.special import logsumexp
import numpy as np
import itertools
import csv

class BinaryCLT:
    def __init__(self, data, root=None, alpha=0.01):
        self.data = np.asarray(data, dtype=float)
        self.N, self.D = self.data.shape  # N = number of samples |D|, D = number of RVs
        self.alpha = alpha
        
        # Handle the root assignment
        if root is None:
            self.root = np.random.randint(self.D)
        else:
            self.root = root
            
        # Compute Mutual Information Matrix
        M = np.zeros((self.D, self.D))
        
        for i in range(self.D):
            for j in range(i + 1, self.D):
                mi = 0.0
                # Tichai prez all binary combinations (y,z)
                for y in [0., 1.]:
                    for z in [0., 1.]:
                        # Count occurrences
                        count_joint = np.sum((self.data[:, i] == y) & (self.data[:, j] == z))
                        count_y = np.sum(self.data[:, i] == y)
                        count_z = np.sum(self.data[:, j] == z)
                        
                        # Apply the exact Laplace correction formulas
                        p_yz = (self.alpha + count_joint) / (4 * self.alpha + self.N)
                        p_y = (2 * self.alpha + count_y) / (4 * self.alpha + self.N)
                        p_z = (2 * self.alpha + count_z) / (4 * self.alpha + self.N)
                        
                        # Add to Mutual Information (I = sum p(y,z) * log(p(y,z) / (p(y)p(z))))
                        if p_yz > 0:  # Prevent log(0) errors
                            mi += p_yz * np.log(p_yz / (p_y * p_z))
                
                # The MI matrix is symmetric
                M[i, j] = mi
                M[j, i] = mi

        # Find MST
        mst = minimum_spanning_tree(-M)
        
        # Direct the tree from the root using Breadth-First Search
        # breadth_first_order returns the order of nodes and an array of predecessors (ei taka shtoto moga)
        node_order, predecessors = breadth_first_order(mst, i_start=self.root, directed=False)
        
        # Format the predecessors array: scipy sets the root's parent to -9999, we need -1
        self.tree = predecessors.copy()
        self.tree[self.tree < 0] = -1
        
        # Convert to a standard Python list of integers to match the expected output
        self.tree = self.tree.astype(int).tolist()

    def get_tree(self):
        return self.tree
    def get_log_params(self):
        ...
    def log_prob(self, x, exhaustive: bool = False):
        ...
    def sample(self, n_samples: int):
        ...