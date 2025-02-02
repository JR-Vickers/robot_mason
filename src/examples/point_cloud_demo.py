"""
Demo script showing point cloud generation from multiple camera views.
"""

import sys
from pathlib import Path
import numpy as np
import cv2
import open3d as o3d
from time import sleep
import threading
import queue

# Add parent directory to path to import from src
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.simulation.core import SimulationCore
from src.vision.point_cloud import PointCloudProcessor

def main():
    # Initialize simulation
    sim = SimulationCore(gui=True, fps=60)
    sim.setup()
    
    # Load robot and create stone block
    robot_path = str(Path(__file__).parent.parent.parent / "robots/ur5/ur5_generated.urdf")
    sim.load_robot(robot_path, base_position=[0, 0, 0])
    sim.create_stone_block(position=(0.5, 0, 0.5))
    
    # Add cameras at different positions for better coverage
    cameras = {
        "front": {
            "position": [1.5, 0, 1.0],
            "target": [0.5, 0, 0.5]
        },
        "top": {
            "position": [0.5, -0.5, 2.0],
            "target": [0.5, 0, 0.5]
        },
        "side": {
            "position": [0.5, 1.5, 1.0],
            "target": [0.5, 0, 0.5]
        }
    }
    
    # Set up cameras
    width, height = 640, 480
    for name, params in cameras.items():
        sim.add_camera(
            name,
            position=params["position"],
            target=params["target"],
            width=width,
            height=height,
            near_val=0.05,
            far_val=4.0
        )
    
    # Initialize point cloud processor
    pcd_processor = PointCloudProcessor(width, height)
    
    # Create coordinate frame for reference
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    coord_frame.transform(np.array([
        [1, 0, 0, 0.5],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]))
    
    print("Starting point cloud visualization. Close window to exit.")
    
    try:
        while True:
            # Step simulation
            sim.step()
            
            # Collect point clouds from all cameras
            point_clouds = []
            for name, camera in sim.cameras.items():
                view = sim.get_camera_view(name)
                if view is None:
                    continue
                
                # Get camera parameters
                intrinsics, extrinsics = pcd_processor.get_camera_parameters(
                    fov=camera.fov,
                    aspect=camera.aspect,
                    near=camera.near_val,
                    far=camera.far_val,
                    view_matrix=camera.view_matrix,
                    width=width,
                    height=height
                )
                
                # Generate point cloud
                pcd = pcd_processor.depth_to_point_cloud(
                    view["depth"],
                    intrinsics,
                    extrinsics,
                    view["rgb"]
                )
                if len(pcd.points) > 0:  # Only add non-empty clouds
                    point_clouds.append(pcd)
            
            # Merge point clouds and visualize
            if point_clouds:
                merged_pcd = pcd_processor.merge_point_clouds(point_clouds)
                if len(merged_pcd.points) > 0:
                    # Visualize the current frame
                    o3d.visualization.draw_geometries(
                        [merged_pcd, coord_frame],
                        window_name="Point Cloud Visualization",
                        width=1280,
                        height=720,
                        point_show_normal=False,
                        mesh_show_wireframe=False,
                        mesh_show_back_face=True
                    )
                
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        sim.close()

if __name__ == "__main__":
    main() 