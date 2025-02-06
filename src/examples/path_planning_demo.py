"""
Demo of path planning capabilities with collision avoidance.
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
import pybullet as p
import pybullet_data

def reset_to_home(robot_control):
    """Reset robot to a known-safe home position"""
    home_config = np.array([0, -np.pi/2, 0, -np.pi/2, 0, 0, 0])  # Straight up position
    robot_control.set_joint_angles(home_config)
    time.sleep(0.5)
    return home_config

def try_configurations(collision_checker, base_config, num_attempts=10):
    """Try different configurations around a base configuration"""
    # Try varying each joint slightly
    for i in range(num_attempts):
        test_config = base_config.copy()
        # Add small random variations to each joint
        variations = np.random.uniform(-np.pi/6, np.pi/6, len(base_config))
        test_config += variations
        if not collision_checker.check_collision(test_config):
            return test_config
    return None

def main():
    # Initialize PyBullet
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")
    
    # Load UR5 robot higher off the ground
    ur5_urdf_path = str(Path(__file__).parent.parent.parent / "robots" / "ur5" / "ur5_generated.urdf")
    robot_id = p.loadURDF(ur5_urdf_path, [0, 0, 0.2], useFixedBase=True)  # Lifted more off ground
    num_joints = p.getNumJoints(robot_id)
    
    # Print joint info for debugging
    print(f"\nLoaded robot with {num_joints} joints:")
    for i in range(num_joints):
        joint_info = p.getJointInfo(robot_id, i)
        print(f"Joint {i}: {joint_info[1].decode('utf-8')}, type: {joint_info[2]}")
    
    # Initialize robot control
    robot_control = RobotControl(robot_id, num_joints)
    
    # First move to home position
    home_config = reset_to_home(robot_control)
    
    # Add some obstacles - moved even further away
    table_pos = [1.0, 0, 0]  # Moved table further away
    table_orient = p.getQuaternionFromEuler([0, 0, 0])
    p.loadURDF("table/table.urdf", table_pos, table_orient, useFixedBase=True)
    
    box_pos = [1.0, 0.3, 0.7]  # Moved box further away
    box_orient = p.getQuaternionFromEuler([0, 0, 0])
    box_id = p.loadURDF("cube.urdf", box_pos, box_orient, useFixedBase=True, globalScaling=0.1)
    
    # Initialize collision checker with larger margin
    collision_checker = CollisionChecker(robot_control, margin=0.05)
    planner = AStarPlanner(robot_control, collision_checker, step_size=0.15)  # Increased step size
    optimizer = TrajectoryOptimizer(robot_control, collision_checker)
    visualizer = PathVisualizer(robot_control)
    
    # Define start and goal configurations
    start_config = np.array([0, -np.pi/4, 0, -np.pi/2, 0, 0, 0])
    goal_config = np.array([np.pi/2, -np.pi/4, 0, -np.pi/2, 0, 0, 0])
    
    # Check and adjust configurations
    print("\nChecking configurations...")
    
    if collision_checker.check_collision(start_config):
        print("Start configuration is in collision, trying alternatives...")
        safe_start = try_configurations(collision_checker, start_config)
        if safe_start is not None:
            print("Found safe starting configuration!")
            start_config = safe_start
        else:
            print("Using home position as start...")
            start_config = home_config
    
    if collision_checker.check_collision(goal_config):
        print("Goal configuration is in collision, trying alternatives...")
        safe_goal = try_configurations(collision_checker, goal_config)
        if safe_goal is not None:
            print("Found safe goal configuration!")
            goal_config = safe_goal
        else:
            print("Could not find safe goal configuration.")
            print("Try adjusting the goal manually...")
            goal_config = start_config  # Just move slightly as a demo
            goal_config[0] += np.pi/4  # Rotate first joint a bit
    
    # Move to start configuration
    print("\nMoving to start configuration...")
    robot_control.set_joint_angles(start_config)
    time.sleep(1)
    
    print("\nPlanning collision-free path...")
    path = planner.plan_path(start_config, goal_config)
    
    if path is None:
        print("No collision-free path found. Try simpler motion...")
        # Fall back to simple interpolated motion as demo
        path = [start_config, goal_config]
    
    print("Optimizing path...")
    smooth_path = optimizer.optimize_path(path, max_acceleration=0.5, smoothing_factor=0.3)  # Even gentler motion
    timed_path = optimizer.time_parameterize(smooth_path, max_velocity=0.3, max_acceleration=0.5)  # Slower execution
    
    print("Visualizing path...")
    visualizer.visualize_path(smooth_path, show_waypoints=True)
    
    # Print path statistics
    print(f"\nPath statistics:")
    print(f"Path length: {len(path)} configurations")
    print(f"Smoothed path length: {len(smooth_path)} configurations")
    print(f"Total trajectory time: {timed_path[-1][0]:.2f} seconds")
    
    print("\nExecuting trajectory...")
    start_time = time.time()
    collision_count = 0  # Track collision warnings
    
    for t, config in timed_path:
        # Wait until the right time
        while time.time() - start_time < t:
            time.sleep(0.001)
            p.stepSimulation()
            
        # Check if still collision-free
        if collision_checker.check_collision(config):
            collision_count += 1
            print(f"Warning: Potential collision detected! ({collision_count} warnings)")
            if collision_count >= 5:  # If too many collisions, try returning home
                print("Too many collisions, returning to home position...")
                reset_to_home(robot_control)
                break
            continue  # Skip this configuration but continue with next
            
        robot_control.set_joint_angles(config)
        p.stepSimulation()
    
    print("Motion complete! Returning to home position...")
    reset_to_home(robot_control)
    
    # Keep simulation running
    print("\nSimulation running. Press Ctrl+C to exit.")
    while True:
        p.stepSimulation()
        time.sleep(1./240.)

if __name__ == "__main__":
    main()