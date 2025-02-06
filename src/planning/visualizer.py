"""
Visualization utilities for path planning.
"""

import numpy as np
from typing import List, Optional
import pybullet as p

class PathVisualizer:
    def __init__(self, robot_control):
        """
        Initialize path visualizer.
        
        Args:
            robot_control: Robot control interface
        """
        self.robot_control = robot_control
        self._path_lines = []
        self._waypoint_spheres = []
        
    def clear(self):
        """Remove all visualization elements."""
        for line_id in self._path_lines:
            p.removeUserDebugItem(line_id)
        for sphere_id in self._waypoint_spheres:
            p.removeUserDebugItem(sphere_id)
        self._path_lines = []
        self._waypoint_spheres = []
        
    def visualize_path(self, path: List[np.ndarray], 
                      color: List[float] = [0, 1, 0, 1],
                      line_width: float = 2.0,
                      show_waypoints: bool = True):
        """
        Visualize a path in the simulation.
        
        Args:
            path: List of configurations
            color: RGBA color for path visualization
            line_width: Width of path lines
            show_waypoints: Whether to show spheres at waypoints
        """
        self.clear()
        
        if len(path) < 2:
            return
            
        # Draw lines between consecutive configurations
        for i in range(len(path) - 1):
            start_pos = self.robot_control.get_end_effector_position(path[i])
            end_pos = self.robot_control.get_end_effector_position(path[i + 1])
            
            line_id = p.addUserDebugLine(
                start_pos,
                end_pos,
                color[:3],
                lineWidth=line_width
            )
            self._path_lines.append(line_id)
            
            if show_waypoints:
                sphere_id = p.addUserDebugPoints(
                    [start_pos],
                    [[1, 1, 0]],  # Yellow for waypoints
                    pointSize=5.0
                )
                self._waypoint_spheres.append(sphere_id)
                
        # Add final waypoint
        if show_waypoints:
            final_pos = self.robot_control.get_end_effector_position(path[-1])
            sphere_id = p.addUserDebugPoints(
                [final_pos],
                [[1, 0, 0]],  # Red for final point
                pointSize=5.0
            )
            self._waypoint_spheres.append(sphere_id)
            
    def visualize_search_tree(self, nodes: List['Node'],
                            color: List[float] = [0.5, 0.5, 1, 0.3],
                            line_width: float = 1.0):
        """
        Visualize the search tree from A* planning.
        
        Args:
            nodes: List of Node objects from search
            color: RGBA color for tree visualization
            line_width: Width of tree lines
        """
        for node in nodes:
            if node.parent is not None:
                start_pos = self.robot_control.get_end_effector_position(node.parent.config)
                end_pos = self.robot_control.get_end_effector_position(node.config)
                
                line_id = p.addUserDebugLine(
                    start_pos,
                    end_pos,
                    color[:3],
                    lineWidth=line_width
                )
                self._path_lines.append(line_id) 