import numpy as np
import torch

def center_of_mass_vector(buffer: torch.Tensor, mass: torch.Tensor):
    """
    Args:
        buffer (torch.Tensor): tensor containing the embedded vectors [shape = (N, D)]
        mass (torch.Tensor): tensor containing the mass (or weight) of each vector [shape = (N, 1)]
    Returns:
        v (torch.Tensor): tensor of the weighted mean of all embedded vectors [shape = (N, D)]
    """
    return torch.sum(buffer * mass, dim = 0) / torch.sum(mass)


class PCA:
    """
    Principal Component Analysis implementation using either the covariance matrix
    or the Gram matrix approach (for high dimensional data where N << D).
    
    DISCLAIMER: IMPLEMENTED WITH CLAUDE SONNET 3.7
    """
    
    def __init__(self, n_components=None, use_gram=False):
        """
        Initialize the PCA object.
        
        Args:
            n_components (int or None): Number of principal components to keep. If None, keep all components.
            
            use_gram (bool): Whether to use the Gram matrix approach (True) or the covariance matrix approach (False).
            
        """
        self.n_components = n_components
        self.use_gram = use_gram
        self.components_ = None
        self.explained_variance_ = None
        self.mean_ = None
        
    def fit(self, X):
        """
        Fit the PCA model to the data.
        
        Args:
            X (torch.Tensor or numpy.ndarray): Training data.
            
        Returns:
            self (object): Returns self.
        """
        # Convert to torch tensor if numpy array
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        
        # Store dimensions
        n_samples, n_features = X.shape
        
        # Center the data
        self.mean_ = torch.mean(X, dim=0)
        X_centered = X - self.mean_
        
        if self.use_gram:
            # Gram matrix approach (for N << D)
            # Compute the Gram matrix: X·X^T (shape: n_samples × n_samples)
            gram_matrix = torch.mm(X_centered, X_centered.t())
            
            # Perform eigendecomposition on the Gram matrix
            eigenvalues, eigenvectors = torch.linalg.eigh(gram_matrix)
            
            # Sort eigenvalues and eigenvectors in descending order
            idx = torch.argsort(eigenvalues, descending=True)
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            # Scale eigenvalues (divide by n_samples - 1)
            eigenvalues = eigenvalues / (n_samples - 1)
            
            # Compute principal components (eigenvectors of covariance matrix) from Gram matrix eigenvectors
            # v_i = X^T·u_i / sqrt(λ_i)
            components = []
            for i in range(len(eigenvalues)):
                if eigenvalues[i] > 1e-10:  # Avoid division by zero
                    v_i = torch.mm(X_centered.t(), eigenvectors[:, i:i+1]) / torch.sqrt(eigenvalues[i] * (n_samples - 1))
                    components.append(v_i.t())
            
            self.components_ = torch.cat(components, dim=0)
            self.explained_variance_ = eigenvalues[:len(components)]
            
        else:
            # Standard approach using covariance matrix
            # Compute the covariance matrix: X^T·X / (n_samples - 1) (shape: n_features × n_features)
            cov_matrix = torch.mm(X_centered.t(), X_centered) / (n_samples - 1)
            
            # Perform eigendecomposition on the covariance matrix
            eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)
            
            # Sort eigenvalues and eigenvectors in descending order
            idx = torch.argsort(eigenvalues, descending=True)
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            self.components_ = eigenvectors.t()  # Shape: (n_components, n_features)
            self.explained_variance_ = eigenvalues
        
        # Limit the number of components if specified
        if self.n_components is not None:
            self.components_ = self.components_[:self.n_components]
            self.explained_variance_ = self.explained_variance_[:self.n_components]
            
        return self
    
    def transform(self, X):
        """
        Apply dimensionality reduction to X.
        
        Args:
            X (torch.Tensor or numpy.ndarray): New data.
            
        Returns:
            X_new (torch.Tensor): Transformed data.
        """
        # Convert to torch tensor if numpy array
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
            
        # Center the data
        X_centered = X - self.mean_
        
        # Project the data onto the principal components
        X_transformed = torch.mm(X_centered, self.components_.t())
        
        return X_transformed
    
    def fit_transform(self, X):
        """
        Fit the model with X and apply the dimensionality reduction on X.
        
        Args:
            X (torch.Tensor or numpy.ndarray): Training data.
            
        Returns:
            X_new (torch.Tensor): Transformed data.
        """
        self.fit(X)
        return self.transform(X)