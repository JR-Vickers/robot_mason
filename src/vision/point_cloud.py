import numpy as np
import open3d as o3d
from typing import Dict, List, Optional, Tuple
import cv2

class PointCloudProcessor:
    def __init__(self, width: int, height: int):
        """Initialize the point cloud processor."""
        self.width = width
        self.height = height
        
        # Create pixel coordinates grid
        self.pixel_coords = np.mgrid[0:height, 0:width].reshape(2, -1)
        self.pixel_coords = np.vstack((self.pixel_coords, np.ones(width * height)))
        
    def depth_to_point_cloud(
        self,
        depth_map: np.ndarray,
        camera_intrinsics: np.ndarray,
        camera_extrinsics: np.ndarray,
        rgb_image: Optional[np.ndarray] = None
    ) -> o3d.geometry.PointCloud:
        """Convert depth map to point cloud."""
        print(f"\nDepth map stats - Shape: {depth_map.shape}, Range: {depth_map.min():.3f} to {depth_map.max():.3f}")
        
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
        
        # Back-project 2D points to 3D
        # Scale pixel coordinates by depth
        points_cam = pixel_coords * depth
        print(f"Camera space points shape: {points_cam.shape}")
        
        # Convert to camera coordinates
        points_cam = np.linalg.inv(camera_intrinsics) @ points_cam
        
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
        extrinsics[:3, :3] = R.T  # Transpose rotation
        extrinsics[:3, 3] = -R.T @ t  # Apply inverse transform to translation
        
        print(f"\nCamera parameters:")
        print(f"View matrix from PyBullet:\n{view_matrix}")
        print(f"Calculated extrinsics:\n{extrinsics}")
        
        return intrinsics, extrinsics 