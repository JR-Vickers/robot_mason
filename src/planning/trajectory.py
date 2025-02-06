"""
Trajectory optimization and smoothing for robot paths.
"""

import numpy as np
from typing import List, Optional, Tuple
from scipy.interpolate import CubicSpline

class TrajectoryOptimizer:
    def __init__(self, robot_control, collision_checker=None):
        """
        Initialize trajectory optimizer.
        
        Args:
            robot_control: Robot control interface
            collision_checker: Optional collision checking interface
        """
        self.robot_control = robot_control
        self.collision_checker = collision_checker
        
    def optimize_path(self, path: List[np.ndarray], 
                     max_acceleration: float = 1.0,
                     smoothing_factor: float = 0.1) -> List[np.ndarray]:
        """
        Optimize and smooth a path while respecting dynamics constraints.
        
        Args:
            path: List of configurations forming the path
            max_acceleration: Maximum allowed joint acceleration
            smoothing_factor: Factor controlling path smoothness (0-1)
            
        Returns:
            Optimized path as list of configurations
        """
        if len(path) < 3:
            return path
            
        # Convert path to array for easier manipulation
        path_array = np.array(path)
        num_points = len(path)
        num_joints = len(path[0])
        
        # Generate time points (assuming constant velocity for now)
        times = np.linspace(0, 1, num_points)
        
        # Create cubic spline for each joint
        splines = []
        for joint in range(num_joints):
            joint_positions = path_array[:, joint]
            spline = CubicSpline(times, joint_positions)
            splines.append(spline)
            
        # Generate smoother trajectory
        num_output_points = int(num_points / smoothing_factor)
        smooth_times = np.linspace(0, 1, num_output_points)
        smooth_path = []
        
        for t in smooth_times:
            # Get position for each joint
            config = np.array([spline(t) for spline in splines])
            
            # Check acceleration limits
            if t > 0 and t < 1:
                accelerations = np.array([spline(t, 2) for spline in splines])
                if np.any(np.abs(accelerations) > max_acceleration):
                    continue
                    
            # Check collision and joint limits
            if (self.collision_checker is None or not self.collision_checker.check_collision(config)) and \
               self.robot_control.is_valid_config(config):
                smooth_path.append(config)
                
        return smooth_path if smooth_path else path
        
    def time_parameterize(self, path: List[np.ndarray], 
                         max_velocity: float = 1.0,
                         max_acceleration: float = 1.0) -> List[Tuple[float, np.ndarray]]:
        """
        Add timing information to path points respecting velocity and acceleration limits.
        
        Args:
            path: List of configurations
            max_velocity: Maximum allowed joint velocity
            max_acceleration: Maximum allowed joint acceleration
            
        Returns:
            List of (time, configuration) tuples
        """
        if len(path) < 2:
            return [(0.0, path[0])] if path else []
            
        timed_path = [(0.0, path[0])]
        current_time = 0.0
        
        for i in range(1, len(path)):
            # Calculate displacement and required time
            displacement = path[i] - path[i-1]
            distance = np.linalg.norm(displacement)
            
            # Time required at max velocity
            min_time = distance / max_velocity
            
            # Time required for acceleration
            accel_time = np.sqrt(2 * distance / max_acceleration)
            
            # Use larger of the two times to ensure constraints are met
            segment_time = max(min_time, accel_time)
            current_time += segment_time
            
            timed_path.append((current_time, path[i]))
            
        return timed_path 