"""
Safety monitoring system for robotic stone carving.

This module provides real-time safety monitoring using point cloud data,
including safety envelope definition and intrusion detection.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
import vtk
from vtk.util import numpy_support

@dataclass
class SafetyEnvelope:
    """Defines a safety envelope around the robot workspace."""
    points: np.ndarray  # Nx3 array of boundary points
    _hull: Optional[vtk.vtkPolyData] = None
    
    def __post_init__(self):
        """Initialize the convex hull from points."""
        if self.points.shape[1] != 3:
            raise ValueError("Points must be Nx3 array")
            
        # Create VTK points
        vtk_points = vtk.vtkPoints()
        vtk_points.SetData(numpy_support.numpy_to_vtk(self.points.astype(np.float32), deep=True))
        
        # Create polydata
        poly = vtk.vtkPolyData()
        poly.SetPoints(vtk_points)
        
        # Create hull
        hull = vtk.vtkDelaunay3D()
        hull.SetInputData(poly)
        hull.Update()
        
        # Extract surface
        surface = vtk.vtkDataSetSurfaceFilter()
        surface.SetInputConnection(hull.GetOutputPort())
        surface.Update()
        
        self._hull = surface.GetOutput()
    
    def is_point_inside(self, point: np.ndarray) -> bool:
        """Check if a point is inside the safety envelope using implicit function."""
        if point.shape != (3,):
            raise ValueError("Point must be 3D vector")
            
        # Create implicit function from hull
        implicit = vtk.vtkImplicitPolyDataDistance()
        implicit.SetInput(self._hull)
        
        # Evaluate point - negative values are inside
        return implicit.EvaluateFunction(point) <= 0

class IntrusionDetector:
    """Detects intrusions into the safety envelope using point cloud data."""
    
    def __init__(self, safety_envelope: SafetyEnvelope, 
                 intrusion_threshold: float = 0.05,
                 min_points_for_intrusion: int = 10):
        self.safety_envelope = safety_envelope
        self.intrusion_threshold = intrusion_threshold
        self.min_points_for_intrusion = min_points_for_intrusion
        
    def process_point_cloud(self, points: np.ndarray) -> Tuple[bool, np.ndarray]:
        """
        Process a point cloud to detect intrusions.
        
        Args:
            points: Nx3 array of points
            
        Returns:
            (has_intrusion, intrusion_points)
        """
        # Simple spatial filtering - take every Nth point
        stride = 5  # Adjust based on performance needs
        filtered_points = points[::stride]
        
        # Check each point for intrusion
        intrusion_mask = np.zeros(len(filtered_points), dtype=bool)
        for i, point in enumerate(filtered_points):
            if self.safety_envelope.is_point_inside(point):
                intrusion_mask[i] = True
        
        intrusion_points = filtered_points[intrusion_mask]
        has_intrusion = len(intrusion_points) >= self.min_points_for_intrusion
        
        return has_intrusion, intrusion_points

def create_default_safety_envelope(
    robot_base: np.ndarray,
    workspace_radius: float,
    height: float
) -> SafetyEnvelope:
    """
    Create a default cylindrical safety envelope around the robot.
    
    Args:
        robot_base: [x, y, z] position of robot base
        workspace_radius: Radius of cylindrical workspace
        height: Height of workspace from base
        
    Returns:
        SafetyEnvelope object
    """
    # Create points forming a cylindrical boundary with more resolution
    num_points = 32  # Increased from 16 for smoother hull
    angles = np.linspace(0, 2*np.pi, num_points, endpoint=False)
    
    # Create bottom and top circles with more intermediate points
    num_layers = 5  # Add intermediate layers
    heights = np.linspace(0, height, num_layers)
    
    circles = []
    for h in heights:
        circle = np.array([
            [workspace_radius * np.cos(theta), workspace_radius * np.sin(theta), h]
            for theta in angles
        ])
        circles.append(circle)
    
    # Add center points for better hull formation
    center_points = np.array([
        [0, 0, h] for h in heights
    ])
    
    # Combine all points and center on robot base
    envelope_points = np.vstack(circles + [center_points]) + robot_base
    
    return SafetyEnvelope(points=envelope_points) 