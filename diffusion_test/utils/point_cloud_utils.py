"""
Point Cloud Utility Functions
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional, Tuple


def generate_sphere_point_cloud(num_points: int = 1024, radius: float = 1.0, noise: float = 0.0):
    """
    Generate a sphere point cloud

    Args:
        num_points: Number of points
        radius: Radius of the sphere
        noise: Amount of Gaussian noise to add

    Returns:
        Point cloud (num_points, 3)
    """
    # Fibonacci sphere algorithm for uniform distribution
    indices = np.arange(0, num_points, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / num_points)
    theta = np.pi * (1 + 5**0.5) * indices

    x = radius * np.cos(theta) * np.sin(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(phi)

    points = np.stack([x, y, z], axis=1)

    if noise > 0:
        points += np.random.randn(*points.shape) * noise

    return points


def generate_cube_point_cloud(num_points: int = 1024, size: float = 1.0):
    """
    Generate a cube point cloud

    Args:
        num_points: Number of points
        size: Size of the cube

    Returns:
        Point cloud (num_points, 3)
    """
    points = np.random.uniform(-size/2, size/2, (num_points, 3))
    return points


def generate_torus_point_cloud(num_points: int = 1024, R: float = 1.0, r: float = 0.3):
    """
    Generate a torus point cloud

    Args:
        num_points: Number of points
        R: Major radius
        r: Minor radius

    Returns:
        Point cloud (num_points, 3)
    """
    u = np.random.uniform(0, 2 * np.pi, num_points)
    v = np.random.uniform(0, 2 * np.pi, num_points)

    x = (R + r * np.cos(v)) * np.cos(u)
    y = (R + r * np.cos(v)) * np.sin(u)
    z = r * np.sin(v)

    points = np.stack([x, y, z], axis=1)
    return points


def normalize_point_cloud(points: np.ndarray, method: str = 'sphere'):
    """
    Normalize point cloud to [-1, 1]

    Args:
        points: Point cloud (N, 3)
        method: 'sphere' or 'box'

    Returns:
        Normalized point cloud
    """
    if method == 'sphere':
        # Normalize to unit sphere
        centroid = np.mean(points, axis=0)
        points = points - centroid
        max_dist = np.max(np.linalg.norm(points, axis=1))
        points = points / max_dist
    elif method == 'box':
        # Normalize to [-1, 1] box
        min_vals = np.min(points, axis=0)
        max_vals = np.max(points, axis=0)
        points = 2 * (points - min_vals) / (max_vals - min_vals) - 1

    return points


def save_point_cloud(points: np.ndarray, filename: str):
    """
    Save point cloud to file

    Args:
        points: Point cloud (N, 3)
        filename: Output filename (.npy or .txt)
    """
    if filename.endswith('.npy'):
        np.save(filename, points)
    else:
        np.savetxt(filename, points)


def load_point_cloud(filename: str):
    """
    Load point cloud from file

    Args:
        filename: Input filename (.npy or .txt)

    Returns:
        Point cloud (N, 3)
    """
    if filename.endswith('.npy'):
        return np.load(filename)
    else:
        return np.loadtxt(filename)


def visualize_point_cloud(
    points: np.ndarray,
    title: str = "Point Cloud",
    save_path: Optional[str] = None,
    show: bool = True,
    elev: float = 30,
    azim: float = 45,
    point_size: float = 1.0
):
    """
    Visualize point cloud using matplotlib

    Args:
        points: Point cloud (N, 3) or tensor (B, N, 3) - will use first batch
        title: Plot title
        save_path: Path to save figure
        show: Whether to display the figure
        elev: Elevation angle
        azim: Azimuth angle
        point_size: Size of points
    """
    # Convert to numpy if tensor
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
        if len(points.shape) == 3:
            points = points[0]  # Take first batch

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
              c=points[:, 2], cmap='viridis', s=point_size, alpha=0.6)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.view_init(elev=elev, azim=azim)

    # Set equal aspect ratio
    max_range = np.array([points[:, 0].max()-points[:, 0].min(),
                         points[:, 1].max()-points[:, 1].min(),
                         points[:, 2].max()-points[:, 2].min()]).max() / 2.0

    mid_x = (points[:, 0].max()+points[:, 0].min()) * 0.5
    mid_y = (points[:, 1].max()+points[:, 1].min()) * 0.5
    mid_z = (points[:, 2].max()+points[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()


def visualize_multiple_point_clouds(
    point_clouds: list,
    titles: list,
    save_path: Optional[str] = None,
    show: bool = True,
    ncols: int = 3
):
    """
    Visualize multiple point clouds in a grid

    Args:
        point_clouds: List of point clouds
        titles: List of titles
        save_path: Path to save figure
        show: Whether to display the figure
        ncols: Number of columns
    """
    n = len(point_clouds)
    nrows = (n + ncols - 1) // ncols

    fig = plt.figure(figsize=(5 * ncols, 5 * nrows))

    for idx, (points, title) in enumerate(zip(point_clouds, titles)):
        # Convert to numpy if tensor
        if isinstance(points, torch.Tensor):
            points = points.detach().cpu().numpy()
            if len(points.shape) == 3:
                points = points[0]

        ax = fig.add_subplot(nrows, ncols, idx + 1, projection='3d')
        ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                  c=points[:, 2], cmap='viridis', s=1.0, alpha=0.6)
        ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()
