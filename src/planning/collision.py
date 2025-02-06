"""
Collision checking utilities for path planning.
"""

import numpy as np
from typing import List, Optional, Tuple
import pybullet as p

class CollisionChecker:
    def __init__(self, robot_control, margin: float = 0.02):
        """
        Initialize collision checker.
        
        Args:
            robot_control: Robot control interface
            margin: Safety margin for collision checking (meters)
        """
        self.robot_control = robot_control
        self.margin = margin
        self._robot_id = robot_control.robot_id
        self._excluded_pairs = self._get_excluded_pairs()
        
    def _get_excluded_pairs(self) -> List[Tuple[int, int]]:
        """Get pairs of link indices that should be excluded from collision checking."""
        excluded = []
        
        # Get number of joints
        num_joints = p.getNumJoints(self._robot_id)
        
        # Add adjacent links to excluded pairs
        for i in range(num_joints):
            link_info = p.getJointInfo(self._robot_id, i)
            parent_idx = link_info[16]  # Parent link index
            if parent_idx >= 0:
                excluded.append((parent_idx, i))
                
        return excluded
        
    def check_self_collision(self, config: np.ndarray) -> bool:
        """
        Check if robot is in self-collision at given configuration.
        
        Args:
            config: Joint configuration to check
            
        Returns:
            True if in self-collision, False otherwise
        """
        # Set robot to configuration
        self.robot_control.set_joint_angles(config)
        
        # Get all link pairs
        num_joints = p.getNumJoints(self._robot_id)
        for i in range(num_joints):
            for j in range(i + 1, num_joints):
                # Skip excluded pairs
                if (i, j) in self._excluded_pairs:
                    continue
                    
                # Check collision between links
                points = p.getClosestPoints(
                    self._robot_id, self._robot_id,
                    distance=self.margin,
                    linkIndexA=i,
                    linkIndexB=j
                )
                
                if points and points[0][8] <= self.margin:  # Distance between objects
                    return True
                    
        return False
        
    def check_environment_collision(self, config: np.ndarray) -> bool:
        """
        Check if robot collides with environment at given configuration.
        
        Args:
            config: Joint configuration to check
            
        Returns:
            True if in collision with environment, False otherwise
        """
        # Set robot to configuration
        self.robot_control.set_joint_angles(config)
        
        # Get all objects in environment
        num_objects = p.getNumBodies()
        
        # Check collision with each object
        for obj_id in range(num_objects):
            if obj_id == self._robot_id:
                continue
                
            # Check collision between robot and object
            points = p.getClosestPoints(
                self._robot_id, obj_id,
                distance=self.margin
            )
            
            if points and points[0][8] <= self.margin:  # Distance between objects
                return True
                
        return False
        
    def check_collision(self, config: np.ndarray) -> bool:
        """
        Check if configuration is in collision (either self or environment).
        
        Args:
            config: Joint configuration to check
            
        Returns:
            True if in collision, False otherwise
        """
        return self.check_self_collision(config) or self.check_environment_collision(config)
        
    def get_closest_obstacle_distance(self, config: np.ndarray) -> float:
        """
        Get distance to closest obstacle from robot at given configuration.
        
        Args:
            config: Joint configuration to check
            
        Returns:
            Distance to closest obstacle in meters
        """
        # Set robot to configuration
        self.robot_control.set_joint_angles(config)
        
        min_distance = float('inf')
        
        # Check distance to all objects
        num_objects = p.getNumBodies()
        for obj_id in range(num_objects):
            if obj_id == self._robot_id:
                continue
                
            points = p.getClosestPoints(
                self._robot_id, obj_id,
                distance=10.0  # Large enough to get meaningful distances
            )
            
            if points:
                min_distance = min(min_distance, points[0][8])
                
        return min_distance 