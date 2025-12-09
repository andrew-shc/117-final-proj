"""
Simple renderer for Gaussian Splatting
"""

import torch
import numpy as np
from pathlib import Path


def qvec_to_rotmat(qvec):
    """Convert quaternion to rotation matrix"""
    q = qvec / np.linalg.norm(qvec)
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ])


def get_projection_matrix(camera, width, height):
    """Create projection matrix from camera intrinsics"""
    fx, fy, cx, cy = camera.params[:4]
    
    # Intrinsic matrix
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    
    return K


def project_points(points, camera, image):
    """Project 3D points to 2D image coordinates"""
    # Get camera pose (world to camera)
    # pycolmap uses cam_from_world() method which returns a Rigid3d transformation
    cam_from_world = image.cam_from_world()
    R = cam_from_world.rotation.matrix()  # rotation matrix
    t = cam_from_world.translation        # translation vector
    
    # Transform points to camera space
    points_cam = (R @ points.T).T + t
    
    # Get projection matrix
    K = get_projection_matrix(camera, camera.width, camera.height)
    
    # Project to image
    points_2d_h = (K @ points_cam.T).T
    points_2d = points_2d_h[:, :2] / points_2d_h[:, 2:3]
    depths = points_2d_h[:, 2]
    
    return points_2d, depths, points_cam


def render_point_cloud(positions, colors, camera, image, width=None, height=None):
    """
    Simple point cloud rendering
    
    Args:
        positions: Nx3 array of 3D positions
        colors: Nx3 array of RGB colors (0-1 range)
        camera: pycolmap Camera object
        image: pycolmap Image object
        width: output width (default: camera width)
        height: output height (default: camera height)
    
    Returns:
        rendered_image: HxWx3 RGB image
    """
    if width is None:
        width = camera.width
    if height is None:
        height = camera.height
    
    # Convert to numpy if tensor
    if torch.is_tensor(positions):
        positions = positions.cpu().numpy()
    if torch.is_tensor(colors):
        colors = colors.cpu().numpy()
    
    # Project points to image
    points_2d, depths, points_cam = project_points(positions, camera, image)
    
    # Create output image
    rendered = np.zeros((height, width, 3))
    depth_buffer = np.full((height, width), np.inf)
    
    # Filter points in front of camera and in bounds
    valid = (depths > 0) & \
            (points_2d[:, 0] >= 0) & (points_2d[:, 0] < width) & \
            (points_2d[:, 1] >= 0) & (points_2d[:, 1] < height)
    
    valid_points = points_2d[valid].astype(int)
    valid_colors = colors[valid]
    valid_depths = depths[valid]
    
    # Render points (simple z-buffer)
    for (x, y), color, depth in zip(valid_points, valid_colors, valid_depths):
        if depth < depth_buffer[y, x]:
            depth_buffer[y, x] = depth
            rendered[y, x] = color
    
    return rendered


def save_image(image, path):
    """Save rendered image"""
    from PIL import Image as PILImage
    
    # Convert to 0-255 range
    image_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    
    # Save
    pil_image = PILImage.fromarray(image_uint8)
    pil_image.save(path)
    print(f"Saved image to {path}")


def render_all_views(trainer, output_dir):
    """
    Render all camera views
    
    Args:
        trainer: GaussianSplattingTrainer object
        output_dir: directory to save rendered images
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Rendering {len(trainer.images)} views...")
    
    for img_id, image in trainer.images.items():
        camera = trainer.cameras[image.camera_id]
        
        # Render
        rendered = render_point_cloud(
            trainer.positions,
            trainer.colors,
            camera,
            image
        )
        
        # Save
        output_path = output_dir / f"{image.name}.png"
        save_image(rendered, output_path)
    
    print(f"Rendered images saved to {output_dir}")
