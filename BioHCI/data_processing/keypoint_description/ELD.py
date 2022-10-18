"""
Created: 2/10/20
© Denisa Qori McDonald 2020 All Rights Reserved
"""
import numpy as np


class ELD:
    """
    Euclidean Levenshtein Distance Metric
    """

    @staticmethod
    def compute_distance(keypress1: np.ndarray, keypress2: np.ndarray) -> float:
        """
        Computes the euclidean levenshtein distance between two tensors. This type of distance is similar to the
        levenshtein distance used on strings, but incorporates euclidean distance instead of the 0 or 1 values used
        for strings. Intuitively it measures the minimal cost of converting one tensor to another.

        Args:
            keypress1 (np.ndarray): the first tensor
            keypress2 (np.ndarray): the second tensor

        Returns:
            minimal_cost (float): the minimal cost of converting one tensor to another; distance between two tensors.

        """
        lev_matrix = np.zeros((keypress1.shape[0], keypress2.shape[0]))
        # changed initial index from 1 to 0 (no idea why I was skipping it before)
        for i in range(0, keypress1.shape[0]):
            for j in range(0, keypress2.shape[0]):

                k1_i = keypress1[i, :]
                k2_j = keypress2[j, :]

                k1_i_norm = np.linalg.norm(k1_i)
                k2_j_norm = np.linalg.norm(k2_j)
                diff_norm = np.linalg.norm(k1_i - k2_j)

                if min(i, j) == 0:
                    lev_matrix[i, j] = max(k1_i_norm, k2_j_norm)

                else:
                    up = lev_matrix[i - 1, j]
                    min_clause_1 = up + k1_i_norm

                    left = lev_matrix[i, j - 1]
                    min_clause_2 = left + k2_j_norm

                    diag = lev_matrix[i - 1, j - 1]
                    min_clause_3 = diag + diff_norm

                    lev_matrix[i, j] = min(min_clause_1, min_clause_2, min_clause_3)

        # return the last element of the diagonal
        minimal_cost = lev_matrix[keypress1.shape[0] - 1, keypress2.shape[0] - 1]
        return minimal_cost

if __name__ == "__main__":
    print("Running ELD module...")

    a = np.array([[1, 3, 0, 2], [4, 11, 2, 3], [7, 1, 10, 0], [5, 2, 6, 8]])
    b = np.array([[2, 1, 5, 12], [3, 4, 9, 1], [19, 7, 2, 6]])
    print (f"ELD (a, b) = {ELD.compute_distance(a, b)}")

    c = np.array([[0, 0, 0], [1, 2, 0]])
    d = np.array([[0, 0, 0], [0, 0, 0]])
    e = np.array([[0, 0, 0], [2, 3, 3]])

    print (f"ELD (c, e) = {ELD.compute_distance(c, e)}\n"
           f"ELD (c, d) + ELD (d, e) = {ELD.compute_distance(c, d) + ELD.compute_distance(d, e)}")
