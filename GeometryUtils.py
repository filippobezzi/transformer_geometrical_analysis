from transformers import GPT2LMHeadModel, GPT2Tokenizer
# External classes
from BufferGeometry import BufferGeometry
from sklearn.decomposition import PCA

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import linalg as la
from numpy.linalg import matrix_rank
import seaborn as sns
import torch
import torch.nn.functional as F

# Initialize tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

"""
GPT-2 Hidden State Geometric Analysis Pipeline

This module provides a comprehensive toolkit for analyzing the geometric properties
of GPT-2 hidden states across different layers and prompts. It implements advanced
geometric analysis techniques including Grassmann manifold distances, volume
computations, and dimensionality reduction for understanding transformer
internal representations.

Key Features:
- Hidden state extraction from pre-trained GPT-2 models
- Grassmann distance computation between layer representations
- Volume analysis of high-dimensional embeddings
- Cosine similarity tracking across layers
- Rank analysis and orthonormal basis extraction
- Comprehensive visualization tools for geometric properties

"""

def get_buffer_states(prompt):
    """
    Extract hidden states from all layers of GPT-2 for a given prompt.
    
    This function tokenizes the input prompt and runs it through the GPT-2 model
    to extract hidden states from all transformer layers. The hidden states
    represent the internal representations at each layer of the network.
    
    INPUT:
    
    -) prompt : str
        Input text string to analyze
        
    RETURNS:
    
    -) buffer_states : numpy.ndarray
        Array of shape (n_blocks, sequence_length, hidden_size) containing
        hidden states for each block, token position, and embedding dimension
    
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids

    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True, return_dict=True)
    
    hidden_states = outputs.hidden_states

    # Extract buffer states
    buffer_states = np.array([layer.squeeze(0).numpy() for layer in hidden_states])

    return buffer_states


def general_analysis(filename):
    """
    Perform comprehensive geometric analysis on multiple prompts from a file.
    
    This function reads prompts from a file and performs extensive geometric
    analysis on their hidden state representations, including volume computation,
    Grassmann distance tracking, cosine similarity analysis, and rank analysis.
    
    The analysis pipeline includes:
    1. Hidden state extraction for each prompt
    2. Dimensionality reduction via SVD (90% variance retention)
    3. Layer-wise geometric property computation
    4. Grassmann distance tracking between consecutive layers
    5. Volume and similarity analysis across the network depth
    (all these functions are inside the class BufferGeometry.py)

    INPUT:
    
    -) filename : str
        Path to text file containing prompts (one per line)
        
    RETURNS:
    --------
    -) volumes : numpy.ndarray
        Array of shape (n_prompts, n_blocks) containing log-volume measurements
        for each prompt and layer
        
    -) gd : numpy.ndarray
        Array of shape (n_prompts, n_blocks) containing Grassmann distances
        between consecutive layers for each prompt
        
    -) cos_sim : numpy.ndarray
        Array of shape (n_prompts, n_blocks) containing cosine similarities
        between each layer's last token and the final layer's last token
        
    -) mean_vector : numpy.ndarray
        Array of shape (n_prompts, n_blocks, hidden_size) containing
        mean vectors for each prompt and layer
        
    -) token_counts : list
        List of word counts for each prompt
        
    -) rank_matrix : dict
        Dictionary mapping prompts to their orthonormal basis matrices
        for rank analysis
        
    Notes:
    ------
    - Performs SVD-based dimensionality reduction to 90% variance retention
    - Uses centered data (mean subtracted) for stable SVD computation
    - Includes NaN and Inf checking for numerical stability
    - Grassmann distance is computed between consecutive reduced representations
    - Rank analysis extracts orthonormal bases via QR decomposition
    """
    
    # Initialize result containers
    volumes = []
    gd = []
    cos_sim = []
    mean_vector = []
    rank_matrix = {}

    # Read prompts from file
    with open(filename, "r") as f:
        prompts = [line.strip() for line in f.readlines() if line.strip()]

    # Loop over the prompts
    for p in prompts:
        # Define matrices to store the results for each prompt
        phrase_volumes = []
        phrase_gd = []
        phrase_cos_sim = []
        phrase_mean_vector = []
        rank_matrix[p] = []
        phrase_rank_matrix = []

        # Get the buffer state 
        buffer_states = get_buffer_states(p)

        # Initial Grassmann reference buffer (from first step)
        buffer_scaled_0 = buffer_states[0,:,:] - np.mean(buffer_states[0,:,:], axis=0, keepdims=True)
        U, S, Vt = la.svd(buffer_scaled_0, lapack_driver='gesvd')
        eigenvalues = S**2
        explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
        n_components = np.argmax(np.cumsum(explained_variance_ratio) >= 0.9)+1

        U_reduced = U[:, :n_components]
        S_reduced = np.diag(S[:n_components])
        Vt_reduced = Vt[:n_components, :]
        buffer_reduced_0 = np.dot(np.dot(U_reduced, S_reduced), Vt_reduced)

        # Check for NaN or Inf values
        assert not np.isnan(buffer_states).any(), "NaNs detected"
        assert not np.isinf(buffer_states).any(), "Infs detected"

        # Loop over buffer steps
        for i in range(buffer_states.shape[0]):
            bg = BufferGeometry(buffer_states[i, :, :])

            # ===== VOLUME =====
            phrase_volumes.append(bg.volume())

            # ===== COSINE SIMILARITY =====
            phrase_cos_sim.append(bg.cosine_similarity(buffer_states[-1, :, :]))

            # ===== MEAN VECTOR =====
            phrase_mean_vector.append(bg.mean_vector())

            # Reduce current buffer
            buffer_scaled = buffer_states[i,:,:] - np.mean(buffer_states[i,:,:], axis=0, keepdims=True)
            U, S, Vt = la.svd(buffer_scaled, lapack_driver='gesvd')
            eigenvalues = S**2
            explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
            n_components = np.argmax(np.cumsum(explained_variance_ratio) >= 0.9)+1

            U_reduced = U[:, :n_components]
            S_reduced = np.diag(S[:n_components])
            Vt_reduced = Vt[:n_components, :]
            buffer_reduced = np.dot(np.dot(U_reduced, S_reduced), Vt_reduced)

            # ===== GRASSMANN DISTANCE =====
            bg_reduced = BufferGeometry(buffer_reduced)
            phrase_gd.append(bg_reduced.grassmann_distance(buffer_reduced_0))

            # Update reference buffer
            buffer_reduced_0 = buffer_reduced

            # ===== RANK MATRIX =====
            Q = bg.extract_Q()
            phrase_rank_matrix = np.vstack(Q)


        # Append results for this phrase
        volumes.append(phrase_volumes)
        gd.append(phrase_gd)
        cos_sim.append(phrase_cos_sim)
        mean_vector.append(phrase_mean_vector)
        rank_matrix[p].append(phrase_rank_matrix)
    
    # Count words in each prompt
    token_counts = [len(tokenizer.encode(prompt)) for prompt in prompts]

    # Convert to numpy arrays
    volumes = np.array(volumes)
    gd = np.array(gd)
    cos_sim = np.array(cos_sim)
    mean_vector = np.array(mean_vector)

    return volumes, gd, cos_sim, mean_vector, token_counts, rank_matrix


def get_heatmap(filename):
    """
    Generate pairwise Grassmann distance heatmaps between all layer pairs.
    
    This function computes a comprehensive distance matrix showing Grassmann
    distances between all possible pairs of layers for each prompt.ù
    
    Procedure:

    1. For each prompt, extract hidden states from all layers
    2. Apply SVD-based dimensionality reduction (90% variance) to each layer
    3. Compute Grassmann distance between every pair of layers
    4. Store results in symmetric distance matrices
    
    INPUT:
    
    -) filename : str
        Path to text file containing prompts (one per line)
        
    RETURNS:
    
    -) gd_heatmap : numpy.ndarray
        Array of shape (n_prompts, n_blocks, n_blocks) containing
        pairwise Grassmann distances between all layer combinations
        
    """
    gd_heatmap = []

    # Read prompts from file
    with open(filename, "r") as f:
        prompts = [line.strip() for line in f.readlines() if line.strip()]

    # Loop over the prompts
    for p in prompts:
        # Define matrices to store the results for each prompt
        single_gd_heatmap = np.zeros((13,13))
        # Get the buffer state 
        buffer_states = get_buffer_states(p)

        # Loop over buffer steps
        for i in range(buffer_states.shape[0]):
            # Initial Grassmann reference buffer (from first step)
            buffer_scaled_0 = buffer_states[i,:,:] - np.mean(buffer_states[i,:,:], axis=0, keepdims=True)
            U, S, Vt = la.svd(buffer_scaled_0, lapack_driver="gesvd")
            eigenvalues = S**2
            explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
            n_components = np.argmax(np.cumsum(explained_variance_ratio) >= 0.9)+1
            
            #Low rank approximation
            U_reduced = U[:, :n_components]
            S_reduced = np.diag(S[:n_components])
            Vt_reduced = Vt[:n_components, :]
            buffer_reduced_0 = np.dot(np.dot(U_reduced, S_reduced), Vt_reduced)
            for j in range(buffer_states.shape[0]):
                bg = BufferGeometry(buffer_states[j, :, :])
                
                # Reduce current buffer
                buffer_scaled = buffer_states[j,:,:] - np.mean(buffer_states[j,:,:], axis=0, keepdims=True)
                U, S, Vt = la.svd(buffer_scaled, lapack_driver="gesvd")
                eigenvalues = S**2
                explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
                n_components = np.argmax(np.cumsum(explained_variance_ratio) >= 0.9)+1
                
                # Low rank approximation   
                U_reduced = U[:, :n_components]
                S_reduced = np.diag(S[:n_components])
                Vt_reduced = Vt[:n_components, :]
                buffer_reduced = np.dot(np.dot(U_reduced, S_reduced), Vt_reduced)

                # ===== GRASSMANN DISTANCE =====
                bg_reduced = BufferGeometry(buffer_reduced)
                single_gd_heatmap[i,j] = bg_reduced.grassmann_distance(buffer_reduced_0)
        
        # append heatmap matrix for that prompt
        gd_heatmap.append(single_gd_heatmap)

    return np.array(gd_heatmap)


# Plotting functions
def plot_geometry(array, word_counts, title, xlabel, ylabel):
    """
    Create line plots showing geometric properties across layers for multiple prompts.
    
    This visualization function generates a multi-line plot where each line
    represents a different prompt, showing how a geometric property (volume,
    Grassmann distance, etc.) changes across network layers.
    
    INPUT:
    
    -) array : numpy.ndarray
        2D array of shape (n_prompts, n_layers) containing values to plot
    -) token_counts : list
        List of token counts for each prompt (used for legend)
    -) title : str
        Plot title
    -) xlabel : str
        X-axis label (typically "Layer")
    -) ylabel : str
        Y-axis label (e.g., "Volume", "Grassmann Distance")
        
    Notes:

    - Supports up to 10 prompts with unique styling
    """
    colormap_name="YlOrRd"
    cmap = plt.colormaps[colormap_name]
    color_positions = np.linspace(0.2, 1, 10)
    colors = [cmap(pos) for pos in color_positions]
    markers = ["^", "o", "s", "X", "v", ">", "<", "P", "D", "*"]
    linestyles = ["solid", "dotted", "dashed", "dashdot", "dashed"]*2


    plt.figure()
    for i in range(array.shape[0]):
        color = colors[i]
        marker = markers[i]
        linestyle = linestyles[i]
        
        plt.plot(range(array.shape[1]), array[i], color=color, marker=marker, linestyle=linestyle, label=f"{word_counts[i]}")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(alpha=0.6)
    plt.legend(loc="best", title="Number of tokens", ncol=3)
    plt.tight_layout()
    plt.show()

def PCA_plot(mean_vector, token_counts):
    """
    Create 2D PCA visualizations of mean vectors across layers for multiple prompts.
    
    This function performs Principal Component Analysis on the mean vectors
    computed for each layer and prompt, creating 2D scatter plots that show
    how the average representation evolves through the network layers.
    
    INPUT:
    
    -) mean_vector : numpy.ndarray
        Array of shape (n_prompts, n_layers, hidden_size) containing
        mean vectors for each prompt and layer
    -) token_counts : list
        List of token counts for each prompt (used in subplot titles)
    
    Notes:

    - PCA is fitted independently for each prompt
    - Colors represent scaled prompt length (layer progression)
    """
    # Create subplots
    fig, axs = plt.subplots(2, 5, figsize=(20, 10), constrained_layout=True)
    axs = axs.flatten()

    num_points = mean_vector.shape[1]
    colors = np.arange(num_points)

    for i in range(10):
        fixed_mean_vector = mean_vector[i, :, :]
        
        pca = PCA(n_components=2)
        proj = pca.fit_transform(fixed_mean_vector)
        
        scatter = axs[i].scatter(proj[:, 0], proj[:, 1], c=colors, cmap="Blues", s=70, edgecolor="black")
        axs[i].set_title(f"Prompt {i+1}, length: {token_counts[i]}")

    # Shared colorbar
    cbar = fig.colorbar(scatter, ax=axs, label="(scaled) prompt length")
    fig.suptitle("2D PCA of of mean vectors", fontsize=16)
    plt.show()

def PCA_plot_3d(mean_vector, token_counts):
    """
    Create 3D PCA visualizations of mean vectors across layers for multiple prompts.
    
    This function performs Principal Component Analysis on the mean vectors
    computed for each layer and prompt, creating 3D scatter plots that show
    how the average representation evolves through the network layers.
    
    INPUT:
    
    -) mean_vector : numpy.ndarray
        Array of shape (n_prompts, n_layers, hidden_size) containing
        mean vectors for each prompt and layer
    -) token_counts : list
        List of word counts for each prompt (used in subplot titles)
    
    Notes:

    - PCA is fitted independently for each prompt
    - Colors represent scaled prompt length (layer progression)
    """
    # Create subplots 
    fig = plt.figure(figsize=(20, 10), constrained_layout=True)
    axs = [fig.add_subplot(2, 5, i+1, projection='3d') for i in range(10)]

    num_points = mean_vector.shape[1]
    colors = np.arange(num_points)

    for i in range(10):
        fixed_mean_vector = mean_vector[i, :, :]
        
        pca = PCA(n_components=3)
        proj = pca.fit_transform(fixed_mean_vector)
        
        scatter = axs[i].scatter(proj[:, 0], proj[:, 1], proj[:, 2], c=colors, cmap="Blues", s=70, edgecolor="black")
        axs[i].set_title(f"Prompt {i+1}, length: {token_counts[i]}")
        axs[i].set_xlabel("PC 1")
        axs[i].set_ylabel("PC 2")
        axs[i].set_zlabel("PC 3")

    # Shared colorbar
    fig.colorbar(scatter, ax=axs, label="(scaled) prompt Length", orientation='vertical')
    fig.suptitle("3D PCA of mean vectors", fontsize=18)
    plt.show()
 

def rank_plot(rank_matrix):
    """
    Plot the effective rank (as fraction of total dimensions) across all analyses.
    
    This function visualizes how much of the total embedding space is effectively
    used by the orthonormal bases extracted from each layer and prompt. The rank
    is normalized by the total embedding dimension (768) to show the fraction
    of space spanned.
    
    INPUT:
    
    -) rank_matrix : dict
        Dictionary mapping prompts to their orthonormal basis matrices
    
    """
    ranks = []
    for prompt, matrices in rank_matrix.items():
        for i, mat in enumerate(matrices):
            rank = matrix_rank(mat)
            ranks.append(rank/768)

    plt.title("Effective Rank per Prompt")
    plt.plot(np.arange(len(ranks)), ranks, marker='s', color = 'red')
    plt.xlabel("Prompts")
    plt.ylabel("Fraction of spanned space")

    plt.grid(alpha=0.6)
    plt.show()

def plot_heatmap(gd_heatmap, token_counts):
    """
    Create a comprehensive heatmap visualization of pairwise Grassmann distances.
    
    This function generates a 2x5 grid of heatmaps, each showing the pairwise
    Grassmann distances between all layer pairs for a specific prompt.

    INPUT:
    
    -) gd_heatmap : numpy.ndarray
        Array of shape (n_prompts, n_blocks, n_blocks) containing
        pairwise Grassmann distances
    
    -) token_counts : numpy.array
        Number of tokens in the prompt (used for the title)
    
    Color Interpretation:
    
    - Dark colors (low values): Similar geometric structure
    - Bright colors (high values): Dissimilar geometric structure
    - Symmetric matrices: distance(i,j) = distance(j,i)
    - Diagonal zeros: each layer is identical to itself
    
    Notes:
    
    - Global normalization ensures consistent color scale across prompts
    """
    # Find global min and max across all heatmaps
    vmin = np.min(gd_heatmap)
    vmax = np.max(gd_heatmap)

    fig, axes = plt.subplots(2, 5, figsize=(25, 8))
    axes = axes.flatten()

    # Create a dummy heatmap to get the mappable for colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])  # [left, bottom, width, height]

    for i in range(gd_heatmap.shape[0]):
        ax = axes[i]
        hm = sns.heatmap(
            gd_heatmap[i],
            ax=ax,
            cmap='YlOrRd',
            vmin=vmin,
            vmax=vmax,
            cbar=False,  
            cbar_ax=None   
        )
        ax.set_title(f"{token_counts[i]} tokens", fontsize = 16)
        ax.tick_params(labelsize=12, width=2, length=6)

    # Create a single colorbar for all subplots
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap="YlOrRd", norm=norm)
    sm.set_array([]) 
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Grassmann distance", fontsize = 15)
    cbar.ax.tick_params(labelsize=12, width=2, length=6)         

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.show()