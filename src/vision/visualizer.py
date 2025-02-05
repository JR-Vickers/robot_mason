"""
Real-time point cloud visualization using VTK.
"""

import vtk
import numpy as np
from vtk.util import numpy_support
import time

def create_coordinate_frame(size=0.5):
    """Create a coordinate frame axes actor."""
    axes = vtk.vtkAxesActor()
    axes.SetTotalLength(size, size, size)
    axes.SetShaftType(0)  # Cylinder shaft
    axes.SetCylinderRadius(0.01)
    return axes

class PointCloudVisualizer:
    def __init__(self, width, height, window_size=(1280, 720)):
        self.width = width
        self.height = height
        
        # Create VTK pipeline for current point cloud
        self.current_points = vtk.vtkPoints()
        self.current_vertices = vtk.vtkCellArray()
        self.current_colors = vtk.vtkUnsignedCharArray()
        self.current_colors.SetNumberOfComponents(3)
        self.current_colors.SetName("Colors")
        
        self.current_polydata = vtk.vtkPolyData()
        self.current_polydata.SetPoints(self.current_points)
        self.current_polydata.SetVerts(self.current_vertices)
        self.current_polydata.GetPointData().SetScalars(self.current_colors)
        
        self.current_mapper = vtk.vtkPolyDataMapper()
        self.current_mapper.SetInputData(self.current_polydata)
        
        self.current_actor = vtk.vtkActor()
        self.current_actor.SetMapper(self.current_mapper)
        self.current_actor.GetProperty().SetPointSize(3)
        
        # Create VTK pipeline for target point cloud (semi-transparent)
        self.target_points = vtk.vtkPoints()
        self.target_vertices = vtk.vtkCellArray()
        self.target_colors = vtk.vtkUnsignedCharArray()
        self.target_colors.SetNumberOfComponents(3)
        self.target_colors.SetName("Colors")
        
        self.target_polydata = vtk.vtkPolyData()
        self.target_polydata.SetPoints(self.target_points)
        self.target_polydata.SetVerts(self.target_vertices)
        self.target_polydata.GetPointData().SetScalars(self.target_colors)
        
        self.target_mapper = vtk.vtkPolyDataMapper()
        self.target_mapper.SetInputData(self.target_polydata)
        
        self.target_actor = vtk.vtkActor()
        self.target_actor.SetMapper(self.target_mapper)
        self.target_actor.GetProperty().SetPointSize(3)
        self.target_actor.GetProperty().SetOpacity(0.3)  # Semi-transparent
        
        # Create text overlay for debug info
        self.text_actor = vtk.vtkTextActor()
        self.text_actor.SetInput("FPS: --\nPoints: --\nCompletion: --%")
        self.text_actor.GetTextProperty().SetFontSize(24)
        self.text_actor.GetTextProperty().SetColor(1.0, 1.0, 1.0)  # White text
        self.text_actor.SetPosition(10, 10)
        
        # Create renderer and window
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.1, 0.1, 0.1)
        self.renderer.AddActor(self.current_actor)
        self.renderer.AddActor(self.target_actor)
        self.renderer.AddActor(self.text_actor)
        self.renderer.AddActor(create_coordinate_frame())
        
        self.window = vtk.vtkRenderWindow()
        self.window.SetSize(*window_size)
        self.window.AddRenderer(self.renderer)
        self.window.SetWindowName("Robot Point Cloud - Live View")
        
        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.window)
        self.interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        
        # Set up default camera view
        camera = self.renderer.GetActiveCamera()
        camera.SetPosition(1, -1, 1)
        camera.SetFocalPoint(0.5, 0, 0.5)
        camera.SetViewUp(0, 0, 1)
        self.renderer.ResetCamera()
        
        # Performance tracking
        self.last_update = time.time()
        self.frame_times = []
        self.update_count = 0
        
    def update_point_cloud(self, cloud_data):
        """Update point cloud visualization with processed data."""
        try:
            points = cloud_data['points']
            colors = cloud_data['colors']
            
            if 'is_target' in cloud_data:
                # Update target point cloud
                self.set_target_cloud(points, colors)
                return
                
            # Update current point cloud
            num_points = len(points)
            
            # Update points
            points_array = points.astype(np.float32)
            self.current_points.SetNumberOfPoints(num_points)
            vtk_array = numpy_support.numpy_to_vtk(points_array, deep=False)
            self.current_points.SetData(vtk_array)
            
            # Update vertices
            if num_points > 0:
                cell_array = np.empty(num_points * 2, dtype=np.int64)
                cell_array[::2] = 1
                cell_array[1::2] = np.arange(num_points, dtype=np.int64)
                cells = numpy_support.numpy_to_vtkIdTypeArray(cell_array, deep=False)
                self.current_vertices.SetCells(num_points, cells)
            
            # Update colors
            if colors is not None:
                color_data = (colors * 255).astype(np.uint8)
                vtk_colors = numpy_support.numpy_to_vtk(color_data, deep=False)
                vtk_colors.SetName("Colors")
                self.current_polydata.GetPointData().SetScalars(vtk_colors)
            
            # Performance tracking
            current_time = time.time()
            elapsed = current_time - self.last_update
            self.frame_times.append(elapsed)
            if len(self.frame_times) > 30:
                self.frame_times.pop(0)
            
            self.update_count += 1
            if self.update_count % 30 == 0:
                avg_fps = 1.0 / (sum(self.frame_times) / len(self.frame_times))
                completion = cloud_data.get('completion', 0.0)
                
                # Update debug overlay
                self.text_actor.SetInput(
                    f"FPS: {avg_fps:.1f}\n"
                    f"Points: {len(points)}\n"
                    f"Completion: {completion:.1f}%"
                )
            
            self.last_update = current_time
            
            # Mark as modified
            self.current_polydata.Modified()
            
        except Exception as e:
            print(f"Error updating point cloud: {str(e)}")
            
    def set_target_cloud(self, points, colors=None):
        """Set the target point cloud for comparison."""
        # Update VTK visualization
        num_points = len(points)
        
        # Update points
        points_array = points.astype(np.float32)
        self.target_points.SetNumberOfPoints(num_points)
        vtk_array = numpy_support.numpy_to_vtk(points_array, deep=False)
        self.target_points.SetData(vtk_array)
        
        # Update vertices
        if num_points > 0:
            cell_array = np.empty(num_points * 2, dtype=np.int64)
            cell_array[::2] = 1
            cell_array[1::2] = np.arange(num_points, dtype=np.int64)
            cells = numpy_support.numpy_to_vtkIdTypeArray(cell_array, deep=False)
            self.target_vertices.SetCells(num_points, cells)
        
        # Update colors
        if colors is not None:
            color_data = (colors * 255).astype(np.uint8)
        else:
            color_data = np.full((num_points, 3), 255, dtype=np.uint8)  # White if no colors
        vtk_colors = numpy_support.numpy_to_vtk(color_data, deep=False)
        vtk_colors.SetName("Colors")
        self.target_polydata.GetPointData().SetScalars(vtk_colors)
        
        self.target_polydata.Modified()

    def start(self, timer_callback, fps=60):
        """Start the visualization with a timer callback."""
        print("\nStarting visualization. Use mouse to interact:")
        print("- Left mouse: Rotate")
        print("- Middle mouse: Pan")
        print("- Right mouse: Zoom")
        print("- Press 'q' to exit")
        
        # Initialize
        self.interactor.Initialize()
        
        # Set up timer for updates
        self.interactor.AddObserver('TimerEvent', timer_callback)
        self.interactor.CreateRepeatingTimer(int(1000/fps))  # Convert fps to milliseconds
        
        # Start interaction
        self.window.Render()
        self.interactor.Start() 