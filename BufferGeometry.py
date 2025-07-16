from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy import linalg as la
import numpy as np

"""
BufferGeometry: class for geometric analysis of high-dimensional data matrices.

This class provides geometric operations and distance computations for analyzing
the structure of data buffers.

Key Features:
- Grassmann manifold distance computation for subspace comparison
- Orthonormal basis extraction via QR decomposition
- Volume computation through singular value analysis
- Token-wise vector extraction and similarity analysis
- Statistical operations like mean vector computation
"""


class BufferGeometry: 
    
    def __init__(self,buffer):
        """
        Initialize BufferGeometry with a data buffer.
        
        INPUT:
        -) buffer : numpy.ndarray
            Input data matrix of shape (sequence_length, embedding_dim)
            representing the geometric structure to analyze.
        """
        self.buffer = buffer


    def grassmann_distance(self, manifold_1):
        """
        Computes the Grassmann geodesic distance between two reduced-rank
        matrices (e.g., after truncated SVD reconstruction).
        
        The Grassmann distance measures the geodesic distance between two subspaces
        on the Grassmann manifold, providing a rotationally invariant metric for
        comparing the geometric structure of different data representations.
        
        This method:
        - Extracts orthonormal bases from both matrices using QR decomposition
        - Computes principal angles between subspaces via SVD of cross-Gram matrix
        - Returns geodesic distance as sqrt(sum of squared principal angles)
        
        INPUT:
        -) manifold_1 : numpy.ndarray
            Second matrix for comparison, same shape as self.buffer
            
        RETURNS:
        -) gd : float
            The Grassmannian geodesic distance between the two subspaces.   
        
        Notes:
        ------
        - Assumes input matrices are not necessarily orthonormal
        - Handles rank-deficient matrices through economic QR decomposition
        - Uses clipping to ensure numerical stability in arccos computation
        """
        X1 = self.buffer
        X2 = manifold_1

        # Orthonormalize the reduced matrices using QR
        Q1, _ = la.qr(X1.T, mode='economic')
        Q2, _ = la.qr(X2.T, mode='economic')

        # Compute the cross-Gram matrix (inner product of orthonormal bases)
        cross_gram = np.dot(Q1.T, Q2)

        # Perform SVD on the cross-Gram matrix
        _, s_values, _ = la.svd(cross_gram, lapack_driver="gesvd")

        # Compute principal angles and Grassmann distance
        angles = np.arccos(np.clip(s_values, -1.0, 1.0)) #clip between cosine range
        gd = np.sqrt(np.sum(angles ** 2))

        return gd

    def extract_Q(self): 
        """
        Extract orthonormal basis from the buffer using QR decomposition.
        
        Performs economic QR decomposition on the transpose of the buffer to obtain
        an orthonormal basis that spans the column space of the original matrix.
        
        RETURNS:
        -)Q : numpy.ndarray
            Orthonormal basis matrix of shape (embedding_dim, rank)
            
        """
        # QR decomposition
        Q, _ = la.qr(self.buffer.T, mode='economic')

        return Q
        
    def volume(self):
        """
        Compute the log-volume of the buffer's geometric structure.
        
        Calculates the logarithmic volume element of the data manifold using
        the product of singular values. 
        
        Procedure:

        - Performs SVD to extract singular values
        - Computes log-volume as sum of log singular values
        
        RETURNS:
    
        -) log_volume : float
            Logarithmic volume of the data manifold
        """
        _, S, _ = la.svd(self.buffer, full_matrices=False)
        S = np.log(S)
        return np.sum(S)
    
    def extract_token_vector(self,token_idx = -1):
        """
        Extract a specific token vector from the buffer.
        
        INPUT:
        
        -) token_idx : int, default=-1
            Index of the token/sample to extract
            Default -1 extracts the last token
            
        RETURNS:
        -) token_vector : numpy.ndarray
            1D array of shape (embedding_dim,) representing the token embedding
        """
        return self.buffer[token_idx,:]
    
    def cosine_similarity(self, manifold, token_idx = -1, manifold_token_idx = -1):
        """
        Compute cosine similarity between token vectors from two manifolds.
        
        Calculates the cosine similarity between specific token embeddings
        from the current buffer and another manifold, providing a measure
        of directional similarity between high-dimensional vectors.
        
        INPUT:
        
        -) manifold : numpy.ndarray
            Second manifold/buffer for comparison
        -) token_idx : int, default=-1
            Token index to extract from current buffer
        -) manifold_token_idx : int, default=-1
            Token index to extract from comparison manifold
            
        RETURNS:
        
        cos_sim : numpy.ndarray
            Cosine similarity values
        """
        last_vector = self.extract_token_vector(token_idx)
        last_vector_manifold = manifold[manifold_token_idx, :]
        cos_sim = cosine_similarity(last_vector.reshape(1, -1), last_vector_manifold.reshape(1, -1))
        return np.array(cos_sim).reshape(-1)
    
    def mean_vector(self):
        """
        Compute the mean vector (centroid) of all tokens in the buffer.
        
        Calculates the element-wise mean across all samples/tokens, providing
        a representative "center" point of the data distribution in the
        embedding space.
        
        RETURNS:
        
        -) mean_vec : numpy.ndarray
            1D array of shape (n_features,) representing the mean embedding
        
        """
        return(np.mean(self.buffer,axis = 0))