"""
Visualization utilities for 3D point clouds.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple


def visualize_point_cloud_matplotlib(
    point_cloud: np.ndarray,
    output_path: Optional[str] = None,
    title: str = "3D Point Cloud",
    point_size: int = 1,
    figsize: Tuple[int, int] = (10, 10)
):
    """
    Visualize point cloud using matplotlib 3D scatter plot.

    Args:
        point_cloud: Point cloud array (N, 3) or (N, 6) with optional RGB
        output_path: Path to save figure (if None, displays interactively)
        title: Plot title
        point_size: Size of points in plot
        figsize: Figure size
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("matplotlib not installed. Install with: pip install matplotlib")
        return

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    coords = point_cloud[:, :3]
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

    if point_cloud.shape[1] >= 6:
        colors = point_cloud[:, 3:6]
        ax.scatter(x, y, z, c=colors, s=point_size, marker='.')
    else:
        ax.scatter(x, y, z, s=point_size, marker='.')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)

    # Set equal aspect ratio
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {output_path}")
        plt.close()
    else:
        plt.show()


def visualize_point_cloud_plotly(
    point_cloud: np.ndarray,
    output_path: Optional[str] = None,
    title: str = "3D Point Cloud",
    point_size: int = 2
):
    """
    Create interactive point cloud visualization using plotly.

    Args:
        point_cloud: Point cloud array (N, 3) or (N, 6) with optional RGB
        output_path: Path to save HTML file (if None, displays in browser)
        title: Plot title
        point_size: Size of points in plot
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("plotly not installed. Install with: pip install plotly")
        return

    coords = point_cloud[:, :3]
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

    if point_cloud.shape[1] >= 6:
        colors = point_cloud[:, 3:6]
        # Convert to RGB strings
        color_strings = [
            f'rgb({int(r*255)},{int(g*255)},{int(b*255)})'
            for r, g, b in colors
        ]
        marker_dict = dict(
            size=point_size,
            color=color_strings,
        )
    else:
        marker_dict = dict(
            size=point_size,
            color=z,
            colorscale='Viridis',
        )

    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=marker_dict
    )])

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        width=800,
        height=800,
    )

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(output_path)
        print(f"Saved interactive visualization to {output_path}")
    else:
        fig.show()


def visualize_point_cloud_open3d(
    point_cloud: np.ndarray,
    window_name: str = "3D Point Cloud"
):
    """
    Visualize point cloud using Open3D (interactive viewer).

    Args:
        point_cloud: Point cloud array (N, 3) or (N, 6) with optional RGB
        window_name: Window title
    """
    try:
        import open3d as o3d
    except ImportError:
        print("open3d not installed. Install with: pip install open3d")
        return

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])

    if point_cloud.shape[1] >= 6:
        pcd.colors = o3d.utility.Vector3dVector(point_cloud[:, 3:6])

    o3d.visualization.draw_geometries(
        [pcd],
        window_name=window_name,
        width=800,
        height=600,
        point_show_normal=False
    )


def save_point_cloud_ply(
    point_cloud: np.ndarray,
    output_path: str,
    binary: bool = False
):
    """
    Save point cloud to PLY file format.

    Args:
        point_cloud: Point cloud array (N, 3) or (N, 6) with optional RGB
        output_path: Path to save PLY file
        binary: Whether to save in binary format (faster, smaller)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    coords = point_cloud[:, :3]
    has_color = point_cloud.shape[1] >= 6

    mode = 'wb' if binary else 'w'
    format_type = 'binary_little_endian' if binary else 'ascii'

    with open(output_path, mode) as f:
        # Write header
        header = f'ply\nformat {format_type} 1.0\n'
        header += f'element vertex {len(coords)}\n'
        header += 'property float x\n'
        header += 'property float y\n'
        header += 'property float z\n'

        if has_color:
            header += 'property uchar red\n'
            header += 'property uchar green\n'
            header += 'property uchar blue\n'

        header += 'end_header\n'

        if binary:
            f.write(header.encode('ascii'))
            # Write binary data
            import struct
            for i in range(len(coords)):
                x, y, z = coords[i]
                f.write(struct.pack('fff', x, y, z))
                if has_color:
                    r, g, b = (point_cloud[i, 3:6] * 255).astype(np.uint8)
                    f.write(struct.pack('BBB', r, g, b))
        else:
            f.write(header)
            # Write ASCII data
            for i in range(len(coords)):
                x, y, z = coords[i]
                if has_color:
                    r, g, b = (point_cloud[i, 3:6] * 255).astype(np.uint8)
                    f.write(f'{x} {y} {z} {r} {g} {b}\n')
                else:
                    f.write(f'{x} {y} {z}\n')

    print(f"Saved point cloud to {output_path}")


def load_point_cloud_ply(file_path: str) -> np.ndarray:
    """
    Load point cloud from PLY file.

    Args:
        file_path: Path to PLY file

    Returns:
        Point cloud array (N, 3) or (N, 6) with optional RGB
    """
    try:
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(file_path)
        points = np.asarray(pcd.points)

        if pcd.has_colors():
            colors = np.asarray(pcd.colors)
            point_cloud = np.concatenate([points, colors], axis=1)
        else:
            point_cloud = points

        print(f"Loaded point cloud from {file_path}: {len(points)} points")
        return point_cloud

    except ImportError:
        print("open3d not available, using manual parsing")
        # Simple PLY parser
        with open(file_path, 'r') as f:
            lines = f.readlines()

        # Parse header
        vertex_count = 0
        has_color = False
        data_start = 0

        for i, line in enumerate(lines):
            if line.startswith('element vertex'):
                vertex_count = int(line.split()[-1])
            elif 'property uchar red' in line or 'property uchar r' in line:
                has_color = True
            elif line.startswith('end_header'):
                data_start = i + 1
                break

        # Parse data
        points = []
        for line in lines[data_start:data_start + vertex_count]:
            values = [float(v) for v in line.strip().split()]
            if has_color:
                x, y, z, r, g, b = values[:6]
                points.append([x, y, z, r/255.0, g/255.0, b/255.0])
            else:
                x, y, z = values[:3]
                points.append([x, y, z])

        point_cloud = np.array(points)
        print(f"Loaded point cloud from {file_path}: {len(points)} points")
        return point_cloud
