from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy import linalg as la
import numpy as np


class BufferGeometry: 
    
    def __init__(self,buffer):
        self.buffer = buffer


    def grassmann_distance(self,manifold_1):
        """ 
        Function that computes the Grassmann distance between 2 manifolds
        of different sizes (using the Cross Gram Matrix)
        """
        #compute cross Gram
        cross_gram = np.dot(self.buffer,manifold_1.T)

        #perform SVD
        _, s_values, _ = la.svd(cross_gram) 

        #extract angles
        angles = np.arccos(np.clip(s_values, -1.0, 1.0))
        gd = np.sqrt(np.sum(angles**2))
        
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
