"""
Test of static point cloud visualization from a single camera view.
"""

import sys
from pathlib import Path
import numpy as np
import open3d as o3d
import pybullet as p
import time

# Add parent directory to path to import from src
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.simulation.core import SimulationCore
from src.vision.point_cloud import PointCloudProcessor

def main():
    # Initialize simulation
    sim = SimulationCore(gui=True, fps=60)
    sim.setup()
    
    # Load robot in initial pose
    robot_path = str(Path(__file__).parent.parent.parent / "robots/ur5/ur5_generated.urdf")
    robot_id = sim.load_robot(robot_path, base_position=[0, 0, 0])
    
    # Add just one camera
    width, height = 640, 480
    sim.add_camera(
        "front",
        position=[1.5, 0, 1.0],
        target=[0.5, 0, 0.5],
        width=width,
        height=height,
        near_val=0.05,
        far_val=4.0
    )
    
    # Initialize point cloud processor
    pcd_processor = PointCloudProcessor(width, height)
    
    # Step simulation once to ensure everything is initialized
    sim.step()
    
    # Get one camera view
    view = sim.get_camera_view("front")
    if view is None:
        print("Failed to get camera view")
        sim.close()
        return
        
    # Get camera parameters
    camera = sim.cameras["front"]
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
    
    print(f"\nGenerated point cloud with {len(pcd.points)} points")
    print(f"Point cloud bounds: [{np.asarray(pcd.points).min():.3f}, {np.asarray(pcd.points).max():.3f}]")
    
    # Create coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    
    print("\nStarting visualization. Close window to exit.")
    
    # Create visualizer and window
    vis = o3d.visualization.Visualizer()
    vis.create_window("Robot Point Cloud - Static View", width=1280, height=720)
    
    # Set render options
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.1, 0.1, 0.1])  # Dark gray
    opt.point_size = 5.0
    opt.light_on = True
    
    # Add geometries
    vis.add_geometry(pcd)
    vis.add_geometry(coord_frame)
    
    # Set up a good default camera view
    vc = vis.get_view_control()
    vc.set_front([1, -1, -1])  # Look from front-top-right
    vc.set_lookat([0.5, 0, 0.5])  # Look at robot
    vc.set_up([0, 0, 1])  # Z-axis up
    vc.set_zoom(0.8)
    
    try:
        # Run visualization loop
        while True:
            # Update visualization
            if not vis.poll_events():
                break
            vis.update_renderer()
            
            # Add a small delay to prevent maxing out CPU
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        vis.destroy_window()
        sim.close()

if __name__ == "__main__":
    main() 