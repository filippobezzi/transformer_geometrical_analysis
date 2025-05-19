from sklearn.preprocessing import StandardScaler
from scipy import linalg as la
import numpy as np

def grassmann_distance(manifold_1,manifold_2):
    """ 
    Function that computes the Grassmann distance between 2 manifolds
    of different sizes (using the Cross Gram Matrix)
    """
    #compute cross Gram
    cross_gram = np.dot(manifold_1,manifold_2.T)

    #perform SVD
    _, s_values, _ = la.svd(cross_gram) 

    #extract angles
    angles = np.arccos(np.clip(s_values, -1.0, 1.0))
    gd = np.sqrt(np.sum(angles**2))
    
    return gd


