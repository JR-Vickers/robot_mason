"""
Base class for robot path planning.
"""

import numpy as np
from typing import List, Tuple, Optional
from abc import ABC, abstractmethod

class PathPlanner(ABC):
    def __init__(self, robot_control, collision_checker=None):
        """
        Initialize path planner.
        
        Args:
            robot_control: Robot control interface
            collision_checker: Optional collision checking interface
        """
        self.robot_control = robot_control
        self.collision_checker = collision_checker
        
    @abstractmethod
    def plan_path(self, start_config: np.ndarray, goal_config: np.ndarray) -> Optional[List[np.ndarray]]:
        """
        Plan a path from start configuration to goal configuration.
        
        Args:
            start_config: Starting joint configuration
            goal_config: Goal joint configuration
            
        Returns:
            List of configurations forming a path, or None if no path found
        """
        pass
    
    def check_collision(self, config: np.ndarray) -> bool:
        """
        Check if a configuration is in collision.
        
        Args:
            config: Joint configuration to check
            
        Returns:
            True if in collision, False otherwise
        """
        if self.collision_checker is None:
            return False
        return self.collision_checker.check_collision(config)
    
    def interpolate_path(self, path: List[np.ndarray], resolution: float = 0.1) -> List[np.ndarray]:
        """
        Interpolate between path points for smooth motion.
        
        Args:
            path: List of configurations
            resolution: Maximum distance between interpolated points
            
        Returns:
            Interpolated path
        """
        if len(path) < 2:
            return path
            
        interpolated_path = []
        for i in range(len(path) - 1):
            start = path[i]
            end = path[i + 1]
            
            # Calculate number of steps needed
            distance = np.linalg.norm(end - start)
            steps = max(2, int(np.ceil(distance / resolution)))
            
            # Linear interpolation
            for t in np.linspace(0, 1, steps):
                config = start + t * (end - start)
                interpolated_path.append(config)
                
        interpolated_path.append(path[-1])
        return interpolated_path
    
    def validate_path(self, path: List[np.ndarray]) -> bool:
        """
        Check if entire path is collision-free.
        
        Args:
            path: List of configurations to check
            
        Returns:
            True if path is valid, False otherwise
        """
        return all(not self.check_collision(config) for config in path) 