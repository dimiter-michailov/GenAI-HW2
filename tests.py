import itertools
import unittest
import numpy as np
from assignment2_task2 import BinaryCLT


class TestBinaryCLT(unittest.TestCase):

    def test_get_tree_example(self):
        # Test the example tree from Task 2a in the assignment description.
        rng = np.random.default_rng(0)
        n = 20000

        x0 = rng.binomial(1, 0.5, n)
        x1 = np.logical_xor(x0, rng.binomial(1, 0.04, n))
        x4 = np.logical_xor(x0, rng.binomial(1, 0.03, n))
        x2 = np.logical_xor(x4, rng.binomial(1, 0.05, n))
        x3 = np.logical_xor(x4, rng.binomial(1, 0.05, n))

        data = np.column_stack([x0, x1, x2, x3, x4]).astype(float)

        model = BinaryCLT(data, root=0, alpha=0.01)

        self.assertEqual(model.get_tree(), [-1, 0, 4, 4, 0])

    def test_get_log_params_assignment_example(self):
        # Test the CPTs from Task 2b in the assignment description.
        tree = [-1, 0, 4, 4, 0]

        probs = np.array([
            [[0.3, 0.7], [0.3, 0.7]],
            [[0.2, 0.8], [0.6, 0.4]],
            [[0.4, 0.6], [0.1, 0.9]],
            [[0.8, 0.2], [0.5, 0.5]],
            [[0.9, 0.1], [0.4, 0.6]],
        ])

        rows, counts = [], []
        for x in itertools.product([0, 1], repeat=5):
            p = (
                probs[0, 0, x[0]]
                * probs[1, x[0], x[1]]
                * probs[2, x[4], x[2]]
                * probs[3, x[4], x[3]]
                * probs[4, x[0], x[4]]
            )
            rows.append(x)
            counts.append(round(100000 * p))

        data = np.repeat(np.array(rows, dtype=float), counts, axis=0)

        model = object.__new__(BinaryCLT)
        model.data = data
        model.N, model.D = data.shape
        model.alpha = 0.0
        model.tree = tree

        np.testing.assert_allclose(
            model.get_log_params(),
            np.log(probs),
            atol=1e-12
        )

    def test_log_prob_sanity_check(self):
        # Sanity check from Task 2c: sum_x p(x) over all fully observed states must be 1.
        data = np.array([
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 1.],
            [0., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.],
            [1., 0., 1.],
            [1., 0., 0.],
        ])

        model = BinaryCLT(data, root=0, alpha=0.01)

        all_states = np.array(
            list(itertools.product([0., 1.], repeat=model.D))
        )

        lp = model.log_prob(all_states, exhaustive=False)

        self.assertAlmostEqual(
            np.sum(np.exp(lp)),
            1.0,
            places=10
        )

    def test_log_prob_sanity_check_exhaustive(self):
        # Same sanity check, but using exhaustive inference.
        data = np.array([
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 1.],
            [0., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.],
            [1., 0., 1.],
            [1., 0., 0.],
        ])

        model = BinaryCLT(data, root=0, alpha=0.01)

        all_states = np.array(
            list(itertools.product([0., 1.], repeat=model.D))
        )

        lp = model.log_prob(all_states, exhaustive=True)

        self.assertAlmostEqual(
            np.sum(np.exp(lp)),
            1.0,
            places=10
        )

    def test_mpe_assignment_example(self):
        # Test the MPE example from Task 2d in the assignment description.
        tree = [-1, 0, 4, 4, 0]

        probs = np.array([
            [[0.3, 0.7], [0.3, 0.7]],
            [[0.2, 0.8], [0.6, 0.4]],
            [[0.4, 0.6], [0.1, 0.9]],
            [[0.8, 0.2], [0.5, 0.5]],
            [[0.9, 0.1], [0.4, 0.6]],
        ])

        rows, counts = [], []

        for x in itertools.product([0, 1], repeat=5):
            p = (
                probs[0, 0, x[0]]
                * probs[1, x[0], x[1]]
                * probs[2, x[4], x[2]]
                * probs[3, x[4], x[3]]
                * probs[4, x[0], x[4]]
            )

            rows.append(x)
            counts.append(round(100000 * p))

        data = np.repeat(np.array(rows, dtype=float), counts, axis=0)

        model = object.__new__(BinaryCLT)
        model.data = data
        model.N, model.D = data.shape
        model.alpha = 0.0
        model.tree = tree

        x = np.array([
            [np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, 1., np.nan, np.nan, 0.]
        ])

        expected = np.array([
            [1, 0, 1, 0, 1],
            [0, 1, 1, 0, 0]
        ])

        np.testing.assert_array_equal(model.mpe(x), expected)

if __name__ == "__main__":
    unittest.main()
