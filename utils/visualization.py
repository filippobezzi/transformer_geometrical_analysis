import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import matplotlib.colors as colors
from IPython.display import HTML

def visualize_particle_evolution(particle_positions, 
                                fps=2, 
                                duration=5, 
                                title="Particle Evolution", 
                                color_map='viridis',
                                trail_length=0,
                                point_size=20,
                                view_angle=(30, 45),
                                save_path=None):
    """
    Create an animation of particles moving in 3D space.

    DISCLAIMER: IMPLEMENTED WITH CLAUDE SONNET 3.7
    
    Args:
        particle_positions (list of torch.Tensor or numpy.ndarray): List of tensors/arrays, each with shape (N, 3) representing N particles' 3D positions at each timestep

        fps (int): Frames per second for the animation

        duration (float): Duration of the animation in seconds
        
        title (str): Title for the animation

        color_map (str): Matplotlib colormap to use for particles
        
        trail_length (int): Number of previous positions to show as trails (0 for no trails)

        point_size (int): Size of the particle points

        view_angle (tuple): Initial viewing angle (elevation, azimuth)
        
        save_path (str or None): Path to save the animation (e.g., 'particle_animation.mp4') or None to not save
    
    Returns:
        animation (Animation object or HTML display):
    """
    # Convert tensor to numpy if needed
    positions = []
    for pos in particle_positions:
        if isinstance(pos, torch.Tensor):
            positions.append(pos.detach().cpu().numpy())
        else:
            positions.append(np.array(pos))
    
    # Get data dimensions
    T = len(positions)  # Number of timesteps
    N = positions[0].shape[0]  # Number of particles
    
    # Create figure and 3D axis
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Set axis labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Find global min and max for consistent axis limits
    all_positions = np.vstack(positions)
    min_val = np.min(all_positions, axis=0)
    max_val = np.max(all_positions, axis=0)
    
    # Add some padding
    padding = (max_val - min_val) * 0.1
    ax.set_xlim(min_val[0] - padding[0], max_val[0] + padding[0])
    ax.set_ylim(min_val[1] - padding[1], max_val[1] + padding[1])
    ax.set_zlim(min_val[2] - padding[2], max_val[2] + padding[2])
    
    # Set initial view angle
    ax.view_init(elev=view_angle[0], azim=view_angle[1])
    
    # Create a scatter plot for particles
    cmap = plt.get_cmap(color_map)
    norm = colors.Normalize(vmin=0, vmax=N-1)
    
    # Initial empty scatter plot
    scatter = ax.scatter([], [], [], s=point_size, c=[], cmap=cmap, norm=norm)
    
    # For trails (if enabled)
    trail_plots = []
    if trail_length > 0:
        for i in range(N):
            # Smaller, more transparent points for trails
            trail, = ax.plot([], [], [], 'o-', markersize=point_size/4, 
                            alpha=0.3, color=cmap(norm(i)))
            trail_plots.append(trail)
    
    def init():
        """Initialize the animation"""
        scatter._offsets3d = ([], [], [])
        scatter.set_array([])
        
        if trail_length > 0:
            for trail in trail_plots:
                trail.set_data([], [])
                trail.set_3d_properties([])
        
        return [scatter] + trail_plots
    
    def update(frame):
        """Update for each animation frame"""
        # Get current positions
        pos = positions[frame]
        
        # Update the scatter plot
        scatter._offsets3d = (pos[:, 0], pos[:, 1], pos[:, 2])
        scatter.set_array(np.arange(N))
        
        # Update trails if enabled
        if trail_length > 0:
            for i in range(N):
                # Calculate trail start (handle beginning of animation)
                trail_start = max(0, frame - trail_length)
                
                # Extract trail positions for this particle
                trail_x = [positions[f][i, 0] for f in range(trail_start, frame + 1)]
                trail_y = [positions[f][i, 1] for f in range(trail_start, frame + 1)]
                trail_z = [positions[f][i, 2] for f in range(trail_start, frame + 1)]
                
                # Update trail
                trail_plots[i].set_data(trail_x, trail_y)
                trail_plots[i].set_3d_properties(trail_z)
        
        # Add frame count to title
        ax.set_title(f"{title} - Frame {frame+1}/{T}")
        
        return [scatter] + trail_plots
    
    # Calculate frames and interval
    frames = T
    interval = 1000 / fps  # milliseconds between frames
    
    # Create animation
    anim = FuncAnimation(fig, update, frames=frames, init_func=init,
                         interval=interval, blit=True)
    
    # Save animation if requested
    if save_path:
        writer = animation.FFMpegWriter(fps=fps)
        anim.save(save_path, writer=writer)
        print(f"Animation saved to {save_path}")
    
    plt.close()  # Close the figure to avoid displaying it twice
    
    # Return animation for display
    return HTML(anim.to_jshtml())