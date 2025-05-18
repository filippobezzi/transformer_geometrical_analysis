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
    
    def __init__(self, use_gram = True, method: str = "fixed_components", n_components = None, threshold = 0.9, ratio = 2):
        """
        Initialize the PCA object.
        
        Args:
            use_gram (bool): Whether to use the Gram matrix approach (True) or the covariance matrix approach (False).

            method ("fixed_components" (default), "fixed_explained_variance", "dropoff", "dropoff_with_theshold", "marchenko_pastur_cutoff"): Method to select how many principal components to keep

            n_components (int or None): Number of principal components to keep. If None, keep all components. Used only if `method = "fixed_components"`
            
            threshold (float): threshold used to determine how much explained variance to keep. Used only if `method = "fixed_explained_variance"`

            ratio (float): Ratio between two consecutive differences in the explained variances per component. Used only if `method = "dropoff"`
            
        """
        self.use_gram = use_gram
        self.method = method
        self.n_components = n_components
        self.threshold = threshold
        self.ratio = ratio

        self.N = None
        self.D = None
        self.eigenvalues = None
        self.eigenvectors = None
        self.components_ = None
        self.explained_variance_ = None
        self.mean_ = None
        return
        
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
        N, D = X.shape
        self.N = N
        self.D = D
        
        # Center the data
        self.mean_ = torch.mean(X, dim=0)
        X_centered = X - self.mean_
        
        if self.use_gram:
            # Gram matrix approach (for N << D)
            # Compute the Gram matrix: X X^T / (N - 1) (shape: N × N)
            gram_matrix = torch.mm(X_centered, X_centered.t()) / (N - 1)
            
            # Perform eigendecomposition on the Gram matrix
            eigenvalues, eigenvectors = torch.linalg.eigh(gram_matrix)
            
            # Sort eigenvalues and eigenvectors in descending order
            idx = torch.argsort(eigenvalues, descending=True)
            self.eigenvalues = eigenvalues[idx]
            self.eigenvectors = eigenvectors[:, idx]
            
            # Compute principal components (eigenvectors of covariance matrix) from Gram matrix eigenvectors
            # v_i = X^T u_i / sqrt( (N-1) λ_i )
            components = []
            # since at least one eigenvalue should be 0 and we have ordered them, we don't consider the last one 
            for i in range(len(self.eigenvalues) - 1):
                # we also avoid any other eigenvals very close to 0
                if self.eigenvalues[i] < 1e-2: break 
                v_i = torch.mm(X_centered.t(), self.eigenvectors[:,i].reshape(-1, 1)) / torch.sqrt(self.eigenvalues[i] * (N - 1))
                components.append(v_i.t())
            
            self.components_ = torch.cat(components, dim=0)
            self.explained_variance_ = self.eigenvalues[:len(components)]
            
        else:
            # Standard approach using covariance matrix
            # Compute the covariance matrix: X^T X / (N - 1) (shape: D × D)
            cov_matrix = torch.mm(X_centered.t(), X_centered) / (N - 1)
            
            # Perform eigendecomposition on the covariance matrix
            eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)
            
            # Sort eigenvalues and eigenvectors in descending order
            idx = torch.argsort(eigenvalues, descending=True)
            self.eigenvalues = eigenvalues[idx]
            self.eigenvectors = eigenvectors[:, idx]
            
            self.components_ = self.eigenvectors.t()  # Shape: (n_components, D)
            self.explained_variance_ = self.eigenvalues
        
        # apply the chosen method to select the principal components 
        self._select_components()
        return self

    def _select_components(self):
        total_explained_variance = torch.sum(self.explained_variance_)

        if (self.method == "fixed_components"):
            self.components_ = self.components_[:self.n_components]
            self.explained_variance_ = self.explained_variance_[:self.n_components]
        
        elif (self.method == "fixed_explained_variance"):
            normalized_explained_variance = self.explained_variance_ / torch.sum(self.explained_variance_)
            self.components_ = self.components_[torch.cumsum(normalized_explained_variance, dim = 0) < self.threshold,:]
            self.explained_variance_ = self.explained_variance_[:len(self.components_)]

        elif (self.method == "dropoff"):
            jumps = self.explained_variance_[:-1] - self.explained_variance_[1:]
            for i in range(len(jumps)-1):
                if (jumps[i] / jumps[i+1] > self.ratio):
                    self.explained_variance_ = self.explained_variance_[:i+1]
                    self.components_ = self.components_[:i+1,:]
                    break

        elif (self.method == "dropoff_with_threshold"):
            jumps = self.explained_variance_[:-1] - self.explained_variance_[1:]
            for i in range(len(jumps)-1):
                if (jumps[i+1] / jumps[i] > self.ratio):
                    if (torch.sum(self.explained_variance_[:i+1]) / total_explained_variance < self.threshold):
                        continue
                    self.explained_variance_ = self.explained_variance_[:i+1]
                    self.components_ = self.components_[:i+1,:]
                    break
        
        elif (self.method == "marchenko_pastur_cutoff"):
            l = self.N / self.D
            var = torch.prod(self.explained_variance_**(1/self.explained_variance_.shape[0]))
            l_plus = (1 + np.sqrt(l))**2 * var
            # l_minus = (1 - np.sqrt(l))**2 * var
            
            # idx = torch.logical_or((self.explained_variance_ < l_minus),(self.explained_variance_ > l_plus))
            idx = (self.explained_variance_ > l_plus)
            self.components_ = self.components_[idx]
            self.explained_variance_ = self.explained_variance_[idx]

        return
    
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