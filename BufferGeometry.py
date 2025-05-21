from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy import linalg as la
import numpy as np


class BufferGeometry: 
    
    def __init__(self,buffer):
        self.buffer = buffer


    def grassmann_distance(self, manifold_1):
        """
        Computes the Grassmann geodesic distance between two reduced-rank
        matrices (e.g., after truncated SVD reconstruction).

        Assumes:
        - self.buffer and manifold_1 are already reduced matrices of shape (n, d_reduced)
        - They are not guaranteed to be orthonormal, so we extract orthonormal bases.

        Returns:
        - gd: float
            The Grassmannian geodesic distance.
        """
        X1 = self.buffer
        X2 = manifold_1

        # Orthonormalize the reduced matrices using QR
        Q1, _ = la.qr(X1, mode='economic')
        Q2, _ = la.qr(X2, mode='economic')

        # Compute the cross-Gram matrix
        cross_gram = np.dot(Q1.T, Q2)

        # Perform SVD on the cross-Gram matrix
        _, s_values, _ = la.svd(cross_gram)

        # Compute principal angles and Grassmann distance
        angles = np.arccos(np.clip(s_values, -1.0, 1.0))
        gd = np.sqrt(np.sum(angles ** 2))

        return gd

    """def volume(self):
        gram_matrix = np.dot(self.buffer.T, self.buffer)
        return np.sqrt(la.det(gram_matrix))"""
    
    def volume(self):
        _, S, _ = la.svd(self.buffer, full_matrices=False)
        S = np.log(S)
        return np.sum(S)
    
    def extract_token_vector(self,token_idx = -1):
        return self.buffer[token_idx,:]
    
    def cosine_similarity(self, manifold, token_idx = -1, manifold_token_idx = -1):
        last_vector = self.extract_token_vector(token_idx)
        last_vector_manifold = manifold[manifold_token_idx, :]
        cos_sim = cosine_similarity(last_vector.reshape(1, -1), last_vector_manifold.reshape(1, -1))
        return np.array(cos_sim).reshape(-1)
    
    def mean_vector(self):
        return(np.mean(self.buffer,axis = 0))