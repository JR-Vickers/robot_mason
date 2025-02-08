"""
Base class for robot path planning.
"""

import numpy as np
from typing import List, Tuple, Optional
from abc import ABC, abstractmethod
from enum import Enum

class ToolOrientation(Enum):
    VERTICAL = "vertical"  # Tool points straight down
    NORMAL = "normal"      # Tool aligns with surface normal
    ANGLED = "angled"     # Tool maintains specific angle to surface
    FREE = "free"         # No orientation constraints

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
    def plan_path(self, start_config: np.ndarray, goal_config: np.ndarray,
                 tool_orientation: ToolOrientation = ToolOrientation.VERTICAL,
                 orientation_constraints: Optional[dict] = None) -> Optional[List[np.ndarray]]:
        """
        Plan a path from start configuration to goal configuration.
        
        Args:
            start_config: Starting joint configuration
            goal_config: Goal joint configuration
            tool_orientation: Desired tool orientation mode
            orientation_constraints: Additional constraints for tool orientation
                For ANGLED mode: {'angle': float} - angle in radians from vertical
                For NORMAL mode: {'surface_point': np.ndarray, 'surface_normal': np.ndarray}
            
        Returns:
            List of configurations forming a path, or None if no path found
        """
        pass
    
    def check_orientation_constraints(self, config: np.ndarray, 
                                   tool_orientation: ToolOrientation,
                                   constraints: Optional[dict] = None) -> bool:
        """
        Check if configuration satisfies tool orientation constraints.
        
        Args:
            config: Joint configuration to check
            tool_orientation: Desired tool orientation mode
            constraints: Additional orientation constraints
            
        Returns:
            True if constraints satisfied, False otherwise
        """
        if tool_orientation == ToolOrientation.FREE:
            return True
            
        try:
            # Get current end effector pose
            ee_pos, ee_orn = self.robot_control.get_end_effector_pose(config)
            tool_direction = self.robot_control.get_tool_direction(ee_orn)
            
            # Even more lenient alignment threshold (was 0.8)
            alignment_threshold = 0.7  # cos(45°) ≈ 0.7, allowing up to 45° deviation
            
            if tool_orientation == ToolOrientation.VERTICAL:
                # Check if tool is pointing downward (within tolerance)
                alignment = np.dot(tool_direction, [0, 0, -1])
                return alignment > alignment_threshold
                
            elif tool_orientation == ToolOrientation.NORMAL and constraints:
                surface_normal = constraints.get('surface_normal')
                if surface_normal is not None:
                    # Check alignment with surface normal (within tolerance)
                    alignment = np.dot(tool_direction, surface_normal)
                    return alignment > alignment_threshold
                    
            elif tool_orientation == ToolOrientation.ANGLED and constraints:
                angle = constraints.get('angle', 0.0)
                # Create target direction based on angle from vertical
                target_dir = np.array([np.sin(angle), 0, -np.cos(angle)])
                alignment = np.dot(tool_direction, target_dir)
                return alignment > alignment_threshold
                
            return True
            
        except Exception as e:
            print(f"Warning: Error checking orientation constraints: {e}")
            return True  # Be permissive on errors
    
    def interpolate_orientation(self, start_orn: np.ndarray, end_orn: np.ndarray, 
                              t: float) -> np.ndarray:
        """
        Interpolate between two orientations using spherical linear interpolation.
        
        Args:
            start_orn: Starting orientation (quaternion)
            end_orn: Ending orientation (quaternion)
            t: Interpolation parameter (0 to 1)
            
        Returns:
            Interpolated orientation
        """
        from scipy.spatial.transform import Slerp, Rotation
        key_rots = Rotation.from_quat([start_orn, end_orn])
        key_times = [0, 1]
        slerp = Slerp(key_times, key_rots)
        return slerp([t])[0].as_quat()
    
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