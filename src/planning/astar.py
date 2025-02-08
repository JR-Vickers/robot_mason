"""
A* path planner implementation for robot motion planning.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Set
from heapq import heappush, heappop
from .planner import PathPlanner, ToolOrientation

class Node:
    def __init__(self, config: np.ndarray, g_cost: float = float('inf'), 
                 h_cost: float = 0.0, parent = None):
        self.config = config
        self.g_cost = g_cost  # Cost from start
        self.h_cost = h_cost  # Heuristic cost to goal
        self.f_cost = g_cost + h_cost
        self.parent = parent
        
    def __lt__(self, other):
        return self.f_cost < other.f_cost
        
    def __eq__(self, other):
        return np.allclose(self.config, other.config)
        
    def __hash__(self):
        return hash(tuple(self.config))

class AStarPlanner(PathPlanner):
    def __init__(self, robot_control, collision_checker=None, 
                 step_size: float = 0.1, max_iterations: int = 10000):
        """
        Initialize A* planner.
        
        Args:
            robot_control: Robot control interface
            collision_checker: Collision checking interface
            step_size: Size of steps for exploring configurations
            max_iterations: Maximum number of iterations before giving up
        """
        super().__init__(robot_control, collision_checker)
        self.step_size = step_size
        self.max_iterations = max_iterations
        
    def heuristic(self, config: np.ndarray, goal: np.ndarray) -> float:
        """
        Compute heuristic cost between configurations.
        
        Args:
            config: Current configuration
            goal: Goal configuration
            
        Returns:
            Estimated cost to goal
        """
        return np.linalg.norm(goal - config)
        
    def get_neighbors(self, config: np.ndarray) -> List[np.ndarray]:
        """
        Get neighboring configurations by taking small steps in each joint.
        
        Args:
            config: Current configuration
            
        Returns:
            List of neighboring configurations
        """
        neighbors = []
        try:
            joint_limits = self.robot_control.get_joint_limits()
            for i in range(len(config)):
                for step in [-self.step_size, self.step_size]:
                    neighbor = config.copy()
                    neighbor[i] += step
                    
                    # Skip if outside joint limits
                    if neighbor[i] < joint_limits[i][0] or neighbor[i] > joint_limits[i][1]:
                        continue
                        
                    neighbors.append(neighbor)
                    
        except (AttributeError, IndexError) as e:
            print(f"Warning: Error generating neighbors: {e}")
            # Fall back to simple neighbor generation without validation
            for i in range(len(config)):
                for step in [-self.step_size, self.step_size]:
                    neighbor = config.copy()
                    neighbor[i] += step
                    neighbors.append(neighbor)
                    
        return neighbors
        
    def reconstruct_path(self, current: Node) -> List[np.ndarray]:
        """
        Reconstruct path from goal node back to start.
        
        Args:
            current: Goal node
            
        Returns:
            List of configurations forming the path
        """
        path = []
        while current is not None:
            path.append(current.config)
            current = current.parent
        return list(reversed(path))
        
    def plan_path(self, start_config: np.ndarray, goal_config: np.ndarray,
                    tool_orientation: ToolOrientation = ToolOrientation.VERTICAL,
                    orientation_constraints: Optional[dict] = None) -> Optional[List[np.ndarray]]:
        """
        Plan path using A* algorithm with orientation constraints.
        
        Args:
            start_config: Starting joint configuration
            goal_config: Goal joint configuration
            tool_orientation: Desired tool orientation mode
            orientation_constraints: Additional orientation constraints
            
        Returns:
            List of configurations forming a path, or None if no path found
        """
        # Check if start and goal satisfy orientation constraints
        if not self.check_orientation_constraints(start_config, tool_orientation, orientation_constraints):
            print("Start configuration violates orientation constraints")
            return None
        
        if not self.check_orientation_constraints(goal_config, tool_orientation, orientation_constraints):
            print("Goal configuration violates orientation constraints")
            return None
        
        # Initialize start node
        start_node = Node(start_config, g_cost=0.0, 
                         h_cost=self.heuristic(start_config, goal_config))
        
        # Initialize open and closed sets
        open_set: List[Node] = [start_node]  # Priority queue
        closed_set: Set[Node] = set()
        
        iterations = 0
        while open_set and iterations < self.max_iterations:
            iterations += 1
            
            # Get node with lowest f_cost
            current = heappop(open_set)
            
            # Check if we reached the goal
            if np.allclose(current.config, goal_config, atol=self.step_size):
                return self.reconstruct_path(current)
                
            closed_set.add(current)
            
            # Explore neighbors
            for neighbor_config in self.get_neighbors(current.config):
                # Skip if in collision or violates orientation constraints
                if (self.check_collision(neighbor_config) or
                    not self.check_orientation_constraints(neighbor_config, 
                                                        tool_orientation,
                                                        orientation_constraints)):
                    continue
                    
                neighbor = Node(neighbor_config)
                if neighbor in closed_set:
                    continue
                    
                # Calculate tentative g_cost
                tentative_g_cost = current.g_cost + self.step_size
                
                # If we haven't seen this neighbor or found a better path
                if neighbor not in open_set or tentative_g_cost < neighbor.g_cost:
                    neighbor.parent = current
                    neighbor.g_cost = tentative_g_cost
                    neighbor.h_cost = self.heuristic(neighbor_config, goal_config)
                    neighbor.f_cost = neighbor.g_cost + neighbor.h_cost
                    
                    if neighbor not in open_set:
                        heappush(open_set, neighbor)
                        
        return None  # No path found 