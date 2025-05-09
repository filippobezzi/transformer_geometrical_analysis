import torch

def mean_vector(buffer):
    """
    Args:
        buffer (torch.Tensor): tensor containing the embedded vectors [shape = (N, D)]
    Returns:
        v (torch.Tensor): tensor of the mean of all embedded vectors [shape = (N, D)]
    """
    return torch.mean(buffer, dim = 0)

def center_of_mass_vector(buffer, mass):
    """
    Args:
        buffer (torch.Tensor): tensor containing the embedded vectors [shape = (N, D)]
        mass (torch.Tensor): tensor containing the mass (or weight) of each vector [shape = (N, 1)]
    Returns:
        v (torch.Tensor): tensor of the weighted mean of all embedded vectors [shape = (N, D)]
    """
    return torch.sum(buffer * mass, dim = 0) / torch.sum(mass)

