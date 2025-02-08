"""
Demo of the safety monitoring system using simulated point cloud data.
"""

import sys
from pathlib import Path
import time
import numpy as np

# Add parent directory to path to import from src
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.vision.visualizer import PointCloudVisualizer
from src.vision.safety_monitor import create_default_safety_envelope, IntrusionDetector

def generate_test_point_cloud(num_points=1000, time_step=0):
    """Generate a test point cloud with some moving points."""
    # Static points in a cylinder
    theta = np.random.uniform(0, 2*np.pi, num_points)
    r = np.random.uniform(0.2, 0.8, num_points)
    z = np.random.uniform(0, 1.0, num_points)
    
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    points = np.column_stack([x, y, z]).astype(np.float32)
    
    # Add some moving points that will intrude into the safety envelope
    num_moving = 50
    t = time_step * 0.05  # Time parameter
    moving_points = np.array([
        [np.cos(t), np.sin(t), 0.5 + 0.2*np.sin(t)],  # Circular motion
        [-0.5 + 0.3*np.sin(t), 0.5 + 0.3*np.cos(t), 0.7],  # Another circular motion
        [0.2, -0.2 + 0.4*np.sin(t), 0.3]  # Up and down motion
    ], dtype=np.float32)
    
    # Replicate moving points with some noise
    noise = np.random.normal(0, 0.05, (num_moving, 3)).astype(np.float32)
    moving_cloud = np.tile(moving_points, (num_moving//3 + 1, 1))[:num_moving] + noise
    
    # Combine static and moving points
    all_points = np.vstack([points, moving_cloud])
    
    # Generate colors (white for static, yellow for moving)
    colors = np.ones((len(all_points), 3), dtype=np.float32)
    colors[num_points:] = np.array([1.0, 1.0, 0.0], dtype=np.float32)  # Yellow for moving points
    
    return {'points': all_points, 'colors': colors}

def main():
    # Create visualizer with larger window for better visibility
    width, height = 320, 240
    vis = PointCloudVisualizer(width, height, window_size=(1280, 720))
    
    # Create safety envelope
    robot_base = np.array([0, 0, 0], dtype=np.float32)
    safety_envelope = create_default_safety_envelope(
        robot_base=robot_base,
        workspace_radius=0.7,  # 0.7 meter radius
        height=1.2  # 1.2 meters height
    )
    
    # Create intrusion detector
    detector = IntrusionDetector(safety_envelope)
    
    # Add safety envelope visualization
    vis.add_safety_envelope(safety_envelope.points)
    
    time_step = 0
    
    def timer_callback(obj, event):
        nonlocal time_step
        try:
            # Generate new point cloud
            cloud_data = generate_test_point_cloud(time_step=time_step)
            time_step += 1
            
            # Check for intrusions
            has_intrusion, intrusion_points = detector.process_point_cloud(cloud_data['points'])
            
            # Update visualization
            vis.update_point_cloud(cloud_data)
            if has_intrusion:
                print(f"WARNING: Intrusion detected with {len(intrusion_points)} points!")
                vis.highlight_intrusion_points(intrusion_points)
        except Exception as e:
            print(f"Error in timer callback: {str(e)}")
    
    # Start visualization
    vis.start(timer_callback)

if __name__ == "__main__":
    main() 