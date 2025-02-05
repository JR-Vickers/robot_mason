import numpy as np
import open3d as o3d
from typing import Dict, List, Optional, Tuple
import cv2
from sklearn.neighbors import NearestNeighbors

class PointCloudProcessor:
    def __init__(self, width: int, height: int, stride: int = 2):
        """Initialize the point cloud processor."""
        self.width = width
        self.height = height
        self.stride = stride  # Skip pixels for faster processing
        
        # Create pixel coordinates grid in (x,y) order with stride
        x = np.arange(0, width, stride)
        y = np.arange(0, height, stride)
        xx, yy = np.meshgrid(x, y)
        self.pixel_coords = np.stack([xx, yy, np.ones_like(xx)], axis=0).reshape(3, -1)
        
    def depth_to_point_cloud(
        self,
        depth_map: np.ndarray,
        camera_intrinsics: np.ndarray,
        camera_extrinsics: np.ndarray,
        rgb_image: Optional[np.ndarray] = None
    ) -> o3d.geometry.PointCloud:
        """Convert depth map to point cloud."""
        print(f"\nDepth map stats - Shape: {depth_map.shape}, Range: {depth_map.min():.3f} to {depth_map.max():.3f}")
        
        # Downsample depth map using stride
        depth_map = depth_map[::self.stride, ::self.stride]
        if rgb_image is not None:
            rgb_image = rgb_image[::self.stride, ::self.stride]
        
        # Reshape depth map to 1D array
        depth = depth_map.reshape(-1)
        
        # Filter out invalid depth values before processing
        valid_mask = (depth > 0) & (depth < np.inf)
        valid_count = np.sum(valid_mask)
        print(f"Valid depth points: {valid_count} out of {len(depth)} ({valid_count/len(depth)*100:.1f}%)")
        
        if not np.any(valid_mask):
            print("Warning: No valid depth points found")
            return o3d.geometry.PointCloud()
            
        depth = depth[valid_mask]
        pixel_coords = self.pixel_coords[:, valid_mask]
        
        # Convert to camera coordinates
        points_cam = np.linalg.inv(camera_intrinsics) @ pixel_coords
        
        # Scale by depth
        points_cam = points_cam * depth[None, :]
        
        # Convert to world coordinates
        points_world = camera_extrinsics @ np.vstack((points_cam, np.ones(points_cam.shape[1])))
        points_world = points_world[:3, :].T
        
        print(f"World space points - Shape: {points_world.shape}, Range: [{points_world.min():.3f}, {points_world.max():.3f}]")
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_world)
        
        # Add colors if RGB image is provided
        if rgb_image is not None:
            rgb_flat = rgb_image.reshape(-1, 3)[valid_mask] / 255.0
            pcd.colors = o3d.utility.Vector3dVector(rgb_flat)
            print(f"Added colors - Range: [{rgb_flat.min():.3f}, {rgb_flat.max():.3f}]")
        
        return pcd
    
    def merge_point_clouds(
        self,
        point_clouds: List[o3d.geometry.PointCloud],
        voxel_size: float = 0.01
    ) -> o3d.geometry.PointCloud:
        """Merge multiple point clouds with optional downsampling."""
        print(f"\nMerging {len(point_clouds)} point clouds")
        
        # Check if we have any non-empty point clouds
        non_empty_clouds = [pcd for pcd in point_clouds if len(pcd.points) > 0]
        if not non_empty_clouds:
            print("Warning: No non-empty point clouds to merge")
            return o3d.geometry.PointCloud()
            
        # Merge all point clouds
        merged_pcd = o3d.geometry.PointCloud()
        total_points = 0
        for i, pcd in enumerate(non_empty_clouds):
            points = np.asarray(pcd.points)
            print(f"Cloud {i}: {len(points)} points, Range: [{points.min():.3f}, {points.max():.3f}]")
            merged_pcd += pcd
            total_points += len(points)
            
        if len(merged_pcd.points) == 0:
            print("Warning: Merged point cloud is empty")
            return merged_pcd
            
        print(f"Total points before downsampling: {total_points}")
        
        # Voxel downsampling to remove duplicates and reduce density
        merged_pcd = merged_pcd.voxel_down_sample(voxel_size)
        print(f"Points after voxel downsampling: {len(merged_pcd.points)}")
        
        # Only do statistical outlier removal if we have enough points
        if len(merged_pcd.points) > 100:
            merged_pcd, _ = merged_pcd.remove_statistical_outlier(
                nb_neighbors=20,
                std_ratio=2.0
            )
            print(f"Points after outlier removal: {len(merged_pcd.points)}")
        
        points = np.asarray(merged_pcd.points)
        print(f"Final point cloud - Points: {len(points)}, Range: [{points.min():.3f}, {points.max():.3f}]")
        return merged_pcd
    
    def get_camera_parameters(
        self,
        fov: float,
        aspect: float,
        near: float,
        far: float,
        view_matrix: np.ndarray,
        width: int,
        height: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert PyBullet camera parameters to intrinsics and extrinsics matrices."""
        # Calculate focal length from FoV
        focal_length = width / (2 * np.tan(fov * np.pi / 360))
        
        # Create camera intrinsics matrix
        intrinsics = np.array([
            [focal_length, 0, width/2],
            [0, focal_length, height/2],
            [0, 0, 1]
        ])
        
        # Convert PyBullet view matrix to camera extrinsics
        # PyBullet gives us the inverse of what we want
        view_matrix = np.array(view_matrix).reshape(4, 4)
        
        # Extract rotation and translation
        R = view_matrix[:3, :3]
        t = view_matrix[:3, 3]
        
        # Create camera extrinsics (transform from world to camera)
        extrinsics = np.eye(4)
        extrinsics[:3, :3] = R  # Use R directly instead of R.T
        extrinsics[:3, 3] = t   # Use t directly instead of -R.T @ t
        
        # Transform from PyBullet's coordinate system to our point cloud coordinate system
        # PyBullet: Z up, -Y forward, X right
        # Point Cloud: Z up, X forward, -Y right
        transform_matrix = np.array([
            [0, -1, 0, 0],  # Point cloud X = -PyBullet Y (forward axis)
            [-1, 0, 0, 0],  # Point cloud Y = -PyBullet X (right axis)
            [0, 0, 1, 0],   # Point cloud Z = PyBullet Z (up axis)
            [0, 0, 0, 1]
        ])
        extrinsics = transform_matrix @ extrinsics
        
        print(f"\nCamera parameters:")
        print(f"View matrix from PyBullet:\n{view_matrix}")
        print(f"Calculated extrinsics:\n{extrinsics}")
        
        return intrinsics, extrinsics

def chamfer_distance(source_points, target_points):
    """
    Compute the Chamfer distance between two point clouds.
    Returns both the mean distance and point-wise distances for visualization.
    """
    # Find nearest neighbors in both directions
    nbrs_source = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(source_points)
    nbrs_target = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(target_points)
    
    # Compute distances from source to target
    distances_source, _ = nbrs_source.kneighbors(target_points)
    distances_target, _ = nbrs_target.kneighbors(source_points)
    
    # Get point-wise distances for visualization
    source_to_target = distances_source.ravel()
    target_to_source = distances_target.ravel()
    
    # Compute mean distance
    mean_distance = (np.mean(source_to_target) + np.mean(target_to_source)) / 2.0
    
    return mean_distance, source_to_target, target_to_source

class PointCloudTracker:
    def __init__(self):
        self.target_cloud = None
        self.current_cloud = None
        self.distance_history = []
        self.completion_threshold = 0.01  # 1cm threshold for considering points as matching
        self.update_counter = 0  # Counter for frame updates
        self.update_frequency = 5  # Only update metrics every N frames
        
    def set_target(self, points, colors=None):
        """Set the target point cloud state."""
        self.target_cloud = {
            'points': points,
            'colors': colors
        }
        
    def update_current(self, points, colors=None):
        """Update the current point cloud state and compute metrics."""
        self.current_cloud = {
            'points': points,
            'colors': colors
        }
        
        # Only compute metrics periodically to save CPU
        self.update_counter += 1
        if self.target_cloud is not None and self.update_counter >= self.update_frequency:
            self.update_counter = 0
            # Downsample points for faster distance computation
            skip = max(1, len(points) // 1000)  # Limit to ~1000 points for comparison
            mean_dist, _, _ = chamfer_distance(
                points[::skip], 
                self.target_cloud['points'][::skip]
            )
            self.distance_history.append(mean_dist)
            
    def get_completion_percentage(self):
        """Estimate completion percentage based on point matching."""
        if self.target_cloud is None or self.current_cloud is None:
            return 0.0
            
        # Downsample points for faster distance computation
        skip = max(1, len(self.current_cloud['points']) // 1000)
        _, source_dists, _ = chamfer_distance(
            self.current_cloud['points'][::skip], 
            self.target_cloud['points'][::skip]
        )
        
        # Count points that are within threshold
        matched_points = np.sum(source_dists < self.completion_threshold)
        total_points = len(source_dists)
        
        return (matched_points / total_points) * 100.0
        
    def get_difference_cloud(self):
        """Generate a point cloud showing differences between current and target."""
        if self.target_cloud is None or self.current_cloud is None:
            return None, None
            
        current_points = self.current_cloud['points']
        target_points = self.target_cloud['points']
        
        # Compute distances with downsampling
        skip = max(1, len(current_points) // 1000)
        _, source_dists, _ = chamfer_distance(
            current_points[::skip], 
            target_points[::skip]
        )
        
        # Interpolate distances back to full resolution
        if skip > 1:
            source_dists = np.repeat(source_dists, skip)[:len(current_points)]
        
        # Normalize distances for coloring
        max_dist = np.max(source_dists)
        if max_dist > 0:
            normalized_dists = source_dists / max_dist
        else:
            normalized_dists = np.zeros_like(source_dists)
        
        # Create color array (red to green based on distance)
        colors = np.zeros((len(current_points), 3))
        colors[:, 0] = normalized_dists  # Red channel
        colors[:, 1] = 1 - normalized_dists  # Green channel
        
        return current_points, colors 