"""
Simple test of Open3D visualization.
"""

import numpy as np
import open3d as o3d
import time

def main():
    # Create a simple cube point cloud
    points = []
    colors = []
    
    # Generate points for each face of a cube
    size = 0.5  # Half-size of cube
    steps = 20  # Points per edge
    
    # Generate a grid of points for each face
    for x in np.linspace(-size, size, steps):
        for y in np.linspace(-size, size, steps):
            # Top and bottom faces (z = ±size)
            points.extend([[x, y, size], [x, y, -size]])
            colors.extend([[1, 0, 0], [1, 0, 0]])  # Red
            
            # Front and back faces (y = ±size)
            points.extend([[x, size, y], [x, -size, y]])
            colors.extend([[0, 1, 0], [0, 1, 0]])  # Green
            
            # Left and right faces (x = ±size)
            points.extend([[size, x, y], [-size, x, y]])
            colors.extend([[0, 0, 1], [0, 0, 1]])  # Blue
    
    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Create coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    
    # Create visualizer and window
    vis = o3d.visualization.Visualizer()
    vis.create_window("Open3D Test", width=1280, height=720)
    
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
    vc.set_lookat([0, 0, 0])   # Look at origin
    vc.set_up([0, 0, 1])       # Z-axis up
    vc.set_zoom(0.8)
    
    print("Visualizer window should be open. Close window to exit.")
    
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

if __name__ == "__main__":
    main() 