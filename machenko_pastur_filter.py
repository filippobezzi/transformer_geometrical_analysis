from sklearn.preprocessing import StandardScaler
from scipy import linalg as la
import numpy as np

def marchenko_pastur_filter(n,d,buffer):
    """
    Function that considers only the meaningful eigenvalues according to Marchenko-Pastur distribution
    """
    ratio = d/n #threshold required from the distribution
    scaler = StandardScaler()
    buffer_scaled = scaler.fit_transform(buffer)
    sigma = 1.0 #variance of the scaled buffer
    #Define the Marchenko Pastur bounds
    lambda_min = sigma * (1 - np.sqrt(ratio))**2
    lambda_max = sigma * (1 + np.sqrt(ratio))**2
    # compute the covariance of the buffer scaled
    cov_buffer = np.cov(buffer_scaled, rowvar=False)
    eigenvalues = la.eigvalsh(cov_buffer)

    return eigenvalues[eigenvalues > lambda_max]

