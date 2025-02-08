"""
Demo of path planning with tool orientation constraints.
"""

import sys
from pathlib import Path
import time
import numpy as np

# Add parent directory to path to import from src
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.simulation.robot_control import RobotControl
from src.planning.astar import AStarPlanner
from src.planning.trajectory import TrajectoryOptimizer
from src.planning.visualizer import PathVisualizer
from src.planning.collision import CollisionChecker
from src.planning.planner import ToolOrientation
import pybullet as p
import pybullet_data

def reset_to_home(robot_control):
    """Reset robot to a known-safe home position"""
    # More natural home position with tool pointing downward
    home_config = np.array([0, -np.pi/3, 0, -2*np.pi/3, 0, np.pi/2, 0])
    robot_control.set_joint_angles(home_config)
    time.sleep(0.5)
    return home_config

def visualize_tool_direction(robot_control, config):
    """Visualize the tool direction for debugging"""
    ee_pos, ee_orn = robot_control.get_end_effector_pose(config)
    tool_dir = robot_control.get_tool_direction(ee_orn)
    end_point = ee_pos + tool_dir * 0.2  # 20cm line
    p.addUserDebugLine(ee_pos, end_point, [1, 0, 0], 2, 0.1)  # Red line showing tool direction

def try_plan_path(planner, start_config, goal_config, orientation_mode, constraints):
    """Try to plan a path with fallbacks"""
    try:
        print(f"Attempting to plan path with {orientation_mode.value} constraints...")
        path = planner.plan_path(start_config, goal_config, 
                               tool_orientation=orientation_mode,
                               orientation_constraints=constraints)
        if path is None:
            print("Initial planning failed, trying with relaxed constraints...")
            path = planner.plan_path(start_config, goal_config,
                                   tool_orientation=ToolOrientation.FREE)
            if path is None:
                print("Failed to plan path even with relaxed constraints")
            else:
                print("Found path with relaxed constraints")
        else:
            print("Successfully found path with original constraints")
        return path
    except Exception as e:
        print(f"Error during path planning: {e}")
        return None

def main():
    # Initialize PyBullet
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")
    
    # Load UR5 robot higher off the ground
    ur5_urdf_path = str(Path(__file__).parent.parent.parent / "robots" / "ur5" / "ur5_generated.urdf")
    robot_id = p.loadURDF(ur5_urdf_path, [0, 0, 0.2], useFixedBase=True)
    num_joints = p.getNumJoints(robot_id)
    
    # Initialize robot control and planners
    robot_control = RobotControl(robot_id, num_joints)
    collision_checker = CollisionChecker(robot_control, margin=0.05)
    planner = AStarPlanner(robot_control, collision_checker, step_size=0.15)
    optimizer = TrajectoryOptimizer(robot_control, collision_checker)
    visualizer = PathVisualizer(robot_control)
    
    # Move to home position
    home_config = reset_to_home(robot_control)
    time.sleep(1)  # Give more time to stabilize
    
    # Add angled surface (represented by a box) - moved directly above base
    surface_pos = [0.5, 0, 0.8]  # Directly above robot base, 80cm up
    surface_orient = p.getQuaternionFromEuler([0, 0, 0])  # No rotation
    surface_id = p.loadURDF("cube.urdf", surface_pos, surface_orient, useFixedBase=True, globalScaling=0.1)  # 10cm cube
    
    # Define surface normal (pointing straight down)
    surface_normal = np.array([0, 0, -1])
    
    print("\nTesting vertical motion until contact:")
    
    # Single test case - vertical motion
    test_cases = [
        # Simple vertical motion using shoulder/elbow
        (ToolOrientation.VERTICAL, None, "Vertical motion until contact",
         np.array([0, -np.pi/6, -np.pi/6, -np.pi/2, 0, 0, 0]))  # More natural upward pose
    ]
    
    for orientation_mode, constraints, description, goal_config in test_cases:
        print(f"\n{'='*50}")
        print(f"Testing: {description}")
        print(f"Goal configuration: {goal_config}")
        print(f"{'='*50}")
        
        # Start from home each time
        start_config = home_config
        
        # Visualize current tool direction
        visualize_tool_direction(robot_control, start_config)
        
        # Try to plan path with fallbacks
        path = try_plan_path(planner, start_config, goal_config, orientation_mode, constraints)
        
        if path is None:
            print(f"Could not find valid path for {orientation_mode.value} orientation, skipping...")
            time.sleep(2)  # Longer pause to show the visualization
            continue
            
        try:
            # Optimize and execute path
            print("Path found! Optimizing...")
            smooth_path = optimizer.optimize_path(path, max_acceleration=0.1, smoothing_factor=0.8)  # Even gentler motion
            timed_path = optimizer.time_parameterize(smooth_path, max_velocity=0.1, max_acceleration=0.1)  # Very slow
            
            print(f"Path length: {len(path)} points")
            print(f"Smoothed path length: {len(smooth_path)} points")
            print(f"Total trajectory time: {timed_path[-1][0]:.2f} seconds")
            
            # Visualize path and goal tool direction
            visualizer.visualize_path(smooth_path, show_waypoints=True)
            visualize_tool_direction(robot_control, goal_config)
            
            print(f"\nExecuting vertical motion until contact...")
            start_time = time.time()
            collision_count = 0
            
            for t, config in timed_path:
                while time.time() - start_time < t:
                    time.sleep(0.001)
                    p.stepSimulation()
                    
                if collision_checker.check_collision(config):
                    print("Contact detected! Stopping motion.")
                    break
                    
                robot_control.set_joint_angles(config)
                p.stepSimulation()
                
            # Hold final position briefly
            print("Holding position...")
            time.sleep(2)
                
        except Exception as e:
            print(f"Error during execution: {e}")
            
        finally:
            # Always try to return to home position
            print("Returning to home position...")
            reset_to_home(robot_control)
            time.sleep(1)
    
    print("\nOrientation planning demo complete!")
    
    # Keep simulation running
    print("\nSimulation running. Press Ctrl+C to exit.")
    while True:
        p.stepSimulation()
        time.sleep(1./240.)

if __name__ == "__main__":
    main() 