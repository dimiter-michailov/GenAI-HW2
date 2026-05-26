from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse.csgraph import breadth_first_order
from scipy.special import logsumexp
import numpy as np
import itertools
import csv

# The datasets worked with are downloaded from the GitHub link
# given in the assignment and placed in the same repository folder.
# Example path: datasets/nltcs/nltcs.train.data
def load_dataset(file_name):
    with open(file_name, "r") as file:
        reader = csv.reader(file, delimiter=",")
        dataset = np.array(list(reader)).astype(float)
    return dataset

class BinaryCLT:
    def __init__(self, data, root=None, alpha=0.01):
        self.data = np.asarray(data, dtype=float)
        self.N, self.D = self.data.shape  # N = number of samples, D = number of RVs
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
                # Go through all binary combinations (y,z)
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
        # breadth_first_order returns the order of nodes and an array of predecessors
        self.node_order, predecessors = breadth_first_order(mst, i_start=self.root, directed=False)
        self.node_order = self.node_order.astype(int).tolist()

        # Format the predecessors array: scipy sets the root's parent to -9999, we want -1
        self.tree = predecessors.copy()
        self.tree[self.tree < 0] = -1

        # Convert to a standard Python list of integers to match the expected output
        self.tree = self.tree.astype(int).tolist()

    def get_tree(self):
        return self.tree

    def get_log_params(self):
        # initialize the table
        log_params = np.zeros((self.D, 2, 2))

        for i in range(self.D):
            parent = self.tree[i]

            # case when node i is the root
            if parent == -1:
                for k in [0, 1]:
                    # we only count the number of occurences for each value of k
                    count_k = np.sum(self.data[:, i] == k)
                    prob = (self.alpha + count_k) / (2 * self.alpha + self.N)

                    # duplicate the probaility (considering the root has no parent)
                    log_params[i, 0, k] = np.log(prob)
                    log_params[i, 1, k] = np.log(prob)

            else:
                # for the non-root nodes, we need to consider the parent node's values
                for j in [0, 1]:
                    count_parent = np.sum(self.data[:, parent] == j)

                    for k in [0, 1]:
                        # count the number of occurences for each combination of parent value j and node value k
                        count_joint = np.sum(
                            (self.data[:, parent] == j) &
                            (self.data[:, i] == k)
                        )

                        # formula with Laplace correction
                        prob = (self.alpha + count_joint) / (
                            2 * self.alpha + count_parent
                        )

                        log_params[i, j, k] = np.log(prob)

        return log_params

    def log_prob(self, x, exhaustive: bool = False):
        log_params = self.get_log_params()
        lp = np.zeros((x.shape[0], 1))

        if exhaustive:
            for n, query in enumerate(x):
                missing = np.where(np.isnan(query))[0]
                all_logps = []

                # We fill the missing RVs with all possible combinations of 0 and 1
                for values in itertools.product([0., 1.], repeat=len(missing)):
                    completed_query = query.copy()
                    completed_query[missing] = values

                    logp = 0.0

                    # Compute log p(x) = sum_i log p(x_i | parent_i)
                    for i in range(self.D):
                        xi = int(completed_query[i])
                        parent = self.tree[i]

                        if parent == -1:
                            # Root case: log p(x_root)
                            logp += log_params[i, 0, xi]
                        else:
                            # Non-root case: log p(x_i | x_parent)
                            xp = int(completed_query[parent])
                            logp += log_params[i, xp, xi]

                    all_logps.append(logp)

                # Compute log p(y) = log sum_z p(y, z)
                lp[n, 0] = logsumexp(all_logps)

        else:
            children = [[] for _ in range(self.D)]
            root = self.root

            # Build child lists from the predecessor list
            for i, parent in enumerate(self.tree):
                if parent != -1:
                    children[parent].append(i)

            for n, query in enumerate(x):
                messages = np.zeros((self.D, 2))

                # Reverse the BFS order to start from the leaves
                for i in reversed(self.node_order):
                    if i == root:
                        continue

                    parent = self.tree[i]

                    for parent_value in [0, 1]:

                        if not np.isnan(query[i]):
                            # Observed Xi: use only the observed value
                            xi = int(query[i])
                            term = log_params[i, parent_value, xi]

                            for child in children[i]:
                                term += messages[child, xi]

                            messages[i, parent_value] = term

                        else:
                            # Missing Xi: sum over xi = 0 and xi = 1
                            terms = []

                            for xi in [0, 1]:
                                term = log_params[i, parent_value, xi]

                                for child in children[i]:
                                    term += messages[child, xi]

                                terms.append(term)

                            messages[i, parent_value] = logsumexp(terms)

                if not np.isnan(query[root]):
                    # If the root is observed: use only the observed root value
                    root_value = int(query[root])
                    logp = log_params[root, 0, root_value]

                    for child in children[root]:
                        logp += messages[child, root_value]

                    lp[n, 0] = logp

                else:
                    # If the root is not observed: sum over root = 0 and root = 1
                    root_terms = []

                    for root_value in [0, 1]:
                        logp = log_params[root, 0, root_value]

                        for child in children[root]:
                            logp += messages[child, root_value]

                        root_terms.append(logp)

                    lp[n, 0] = logsumexp(root_terms)

        return lp

    def mpe(self, x):
        log_params = self.get_log_params()
        x_mpe = np.zeros(x.shape, dtype=int)

        children = [[] for _ in range(self.D)]
        root = self.root

        # Build child lists from the predecessor list
        for i, parent in enumerate(self.tree):
            if parent != -1:
                children[parent].append(i)

        for n, query in enumerate(x):
            messages = np.zeros((self.D, 2))
            choices = np.zeros((self.D, 2), dtype=int)

            # Compute max-product messages going bottom-up
            for i in reversed(self.node_order):
                if i == root:
                    continue

                parent = self.tree[i]

                for parent_value in [0, 1]:

                    if not np.isnan(query[i]):
                        # Observed Xi: use only the observed value
                        xi = int(query[i])
                        term = log_params[i, parent_value, xi]

                        for child in children[i]:
                            term += messages[child, xi]

                        messages[i, parent_value] = term
                        choices[i, parent_value] = xi

                    else:
                        # Missing Xi: maximize over xi = 0 and xi = 1
                        terms = []

                        for xi in [0, 1]:
                            term = log_params[i, parent_value, xi]

                            for child in children[i]:
                                term += messages[child, xi]

                            terms.append(term)

                        best_xi = int(np.argmax(terms))
                        messages[i, parent_value] = terms[best_xi]
                        choices[i, parent_value] = best_xi

            # Choose the max-product root value
            if not np.isnan(query[root]):
                root_value = int(query[root])
            else:
                root_terms = []

                for root_value_candidate in [0, 1]:
                    term = log_params[root, 0, root_value_candidate]

                    for child in children[root]:
                        term += messages[child, root_value_candidate]

                    root_terms.append(term)

                root_value = int(np.argmax(root_terms))

            # Top-down pass: reconstruct the best joint assignment
            assignment = np.zeros(self.D, dtype=int)
            assignment[root] = root_value

            for node in self.node_order:
                for child in children[node]:
                    parent_value = assignment[node]
                    assignment[child] = choices[child, parent_value]

            x_mpe[n] = assignment

        return x_mpe

    def sample(self, n_samples: int):
        ...

if __name__ == "__main__":
    train_data = load_dataset("datasets/nltcs/nltcs.train.data")
    valid_data = load_dataset("datasets/nltcs/nltcs.valid.data")
    test_data = load_dataset("datasets/nltcs/nltcs.test.data")

    model = BinaryCLT(train_data, alpha=0.01)

    print(model.get_tree())
