"""
Test of real-time point cloud visualization using VTK with multiprocessing.
"""

import sys
from pathlib import Path
import numpy as np
import vtk
from vtk.util import numpy_support
import pybullet as p
import time
from multiprocessing import Process, Queue
import queue

# Add parent directory to path to import from src
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.simulation.core import SimulationCore

def create_coordinate_frame(size=0.5):
    """Create a coordinate frame axes actor."""
    axes = vtk.vtkAxesActor()
    axes.SetTotalLength(size, size, size)
    axes.SetShaftType(0)  # Cylinder shaft
    axes.SetCylinderRadius(0.01)
    return axes

def numpy_to_vtk_points(points, colors=None):
    """Convert numpy arrays to VTK points and colors."""
    # Create VTK points
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_support.numpy_to_vtk(points))
    
    # Create polydata
    point_polydata = vtk.vtkPolyData()
    point_polydata.SetPoints(vtk_points)
    
    # Create vertices (required for point rendering)
    vertices = vtk.vtkCellArray()
    for i in range(len(points)):
        vertices.InsertNextCell(1)
        vertices.InsertCellPoint(i)
    point_polydata.SetVerts(vertices)
    
    # Add colors if provided
    if colors is not None:
        vtk_colors = numpy_support.numpy_to_vtk(colors)
        vtk_colors.SetName("Colors")
        point_polydata.GetPointData().SetScalars(vtk_colors)
    
    return point_polydata

def depth_to_point_cloud(depth_img, intrinsics, extrinsics, rgb_img=None):
    """Convert depth image to point cloud."""
    height, width = depth_img.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    
    # Create pixel coordinates
    x = np.arange(width)
    y = np.arange(height)
    xx, yy = np.meshgrid(x, y)
    
    # Convert to 3D points
    z = depth_img
    x = (xx - cx) * z / fx
    y = (yy - cy) * z / fy
    
    # Stack and reshape
    points = np.stack([x, y, z], axis=-1)
    points = points.reshape(-1, 3)
    
    # Filter out invalid points
    valid_mask = z.reshape(-1) > 0
    points = points[valid_mask]
    
    # Add colors if RGB image is provided
    colors = None
    if rgb_img is not None:
        colors = rgb_img.reshape(-1, 3)[valid_mask] / 255.0
    
    # Transform points using extrinsics
    points = np.dot(points, extrinsics[:3, :3].T) + extrinsics[:3, 3]
    
    return points, colors

class SimulationProcess:
    def __init__(self, data_queue):
        self.data_queue = data_queue
        
    def run(self):
        # Initialize simulation
        sim = SimulationCore(gui=True, fps=60)  # Back to 60 FPS
        sim.setup()
        
        # Configure PyBullet
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
        p.setRealTimeSimulation(1)
        
        # Load robot
        robot_path = str(Path(__file__).parent.parent.parent / "robots/ur5/ur5_generated.urdf")
        robot_id = sim.load_robot(robot_path, base_position=[0, 0, 0])
        
        # Add camera
        width, height = 320, 240
        sim.add_camera(
            "front",
            position=[1.5, 0, 1.0],
            target=[0.5, 0, 0.5],
            width=width,
            height=height,
            near_val=0.05,
            far_val=4.0
        )
        
        # Define joint positions for continuous motion
        joint_positions = [
            [0, -1.0, 1.0, -1.57, -1.57, 0],  # Position 1
            [1.57, -0.5, 0.5, -0.5, -1.57, 0],  # Position 2
            [3.14, -1.0, 1.0, -1.57, -1.57, 0],  # Position 3
            [-1.57, -0.5, 0.5, -0.5, -1.57, 0],  # Position 4
        ]
        current_target = 0
        interpolation_steps = 120  # 2 seconds at 60 FPS
        step_counter = 0
        
        print("Simulation running at 60 FPS...")
        last_view_time = time.time()
        
        try:
            while True:
                # Get current and next target positions
                current_pos = joint_positions[current_target]
                next_pos = joint_positions[(current_target + 1) % len(joint_positions)]
                
                # Interpolate between positions
                alpha = step_counter / interpolation_steps
                interpolated_pos = [
                    current_pos[i] + (next_pos[i] - current_pos[i]) * alpha
                    for i in range(len(current_pos))
                ]
                
                # Set joint positions
                for i in range(len(interpolated_pos)):
                    p.setJointMotorControl2(
                        robot_id, i, p.POSITION_CONTROL,
                        targetPosition=interpolated_pos[i],
                        force=100
                    )
                
                # Update counters
                step_counter += 1
                if step_counter >= interpolation_steps:
                    step_counter = 0
                    current_target = (current_target + 1) % len(joint_positions)
                
                # Step simulation at full speed
                sim.step()
                
                # Send camera view at reduced rate (15 FPS)
                current_time = time.time()
                if current_time - last_view_time >= 1.0/15.0:
                    view = sim.get_camera_view("front")
                    if view is not None:
                        # Pack only necessary data
                        camera_data = {
                            'view_matrix': np.array(sim.cameras["front"].view_matrix),
                            'depth': view["depth"],
                            'rgb': view["rgb"]
                        }
                        try:
                            # Non-blocking put with timeout
                            self.data_queue.put(camera_data, timeout=0.001)
                        except queue.Full:
                            # Skip frame if queue is full
                            pass
                    last_view_time = current_time
                
        except KeyboardInterrupt:
            print("\nClosing simulation...")
        finally:
            sim.close()

class PointCloudVisualizer:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        
        # Calculate static camera parameters
        fov = 60.0  # Default FOV
        self.intrinsics = np.array([
            [width / (2 * np.tan(fov * np.pi / 360)), 0, width/2],
            [0, height / (2 * np.tan(fov * np.pi / 360)), height/2],
            [0, 0, 1]
        ])
        
        # Create VTK pipeline
        self.points = vtk.vtkPoints()
        self.vertices = vtk.vtkCellArray()
        self.colors = vtk.vtkUnsignedCharArray()
        self.colors.SetNumberOfComponents(3)
        self.colors.SetName("Colors")
        
        self.point_polydata = vtk.vtkPolyData()
        self.point_polydata.SetPoints(self.points)
        self.point_polydata.SetVerts(self.vertices)
        self.point_polydata.GetPointData().SetScalars(self.colors)
        
        self.mapper = vtk.vtkPolyDataMapper()
        self.mapper.SetInputData(self.point_polydata)
        
        self.actor = vtk.vtkActor()
        self.actor.SetMapper(self.mapper)
        self.actor.GetProperty().SetPointSize(3)
        
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.1, 0.1, 0.1)
        self.renderer.AddActor(self.actor)
        self.renderer.AddActor(create_coordinate_frame())
        
        self.window = vtk.vtkRenderWindow()
        self.window.SetSize(1280, 720)
        self.window.AddRenderer(self.renderer)
        self.window.SetWindowName("Robot Point Cloud - Live View")
        
        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.window)
        self.interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        
        camera = self.renderer.GetActiveCamera()
        camera.SetPosition(1, -1, 1)
        camera.SetFocalPoint(0.5, 0, 0.5)
        camera.SetViewUp(0, 0, 1)
        self.renderer.ResetCamera()
        
        # Performance tracking
        self.last_update = time.time()
        self.frame_times = []
        self.update_count = 0
    
    def update_point_cloud(self, camera_data):
        try:
            # Update view matrix and compute extrinsics
            view_matrix = camera_data['view_matrix'].reshape(4, 4)
            extrinsics = np.linalg.inv(view_matrix)
            
            # Generate point cloud
            points, colors = depth_to_point_cloud(
                camera_data['depth'],
                self.intrinsics,
                extrinsics,
                camera_data['rgb']
            )
            
            # Batch update points and vertices
            num_points = len(points)
            
            # Update points (use SetVoidArray for faster updates)
            points_array = points.astype(np.float32)
            self.points.SetNumberOfPoints(num_points)
            vtk_array = numpy_support.numpy_to_vtk(points_array, deep=False)
            self.points.SetData(vtk_array)
            
            # Update vertices efficiently
            if num_points > 0:
                cell_array = np.empty(num_points * 2, dtype=np.int64)
                cell_array[::2] = 1  # vertex count
                cell_array[1::2] = np.arange(num_points, dtype=np.int64)  # vertex index
                cells = numpy_support.numpy_to_vtkIdTypeArray(cell_array, deep=False)
                self.vertices.SetCells(num_points, cells)
            
            # Update colors efficiently
            if colors is not None:
                color_data = (colors * 255).astype(np.uint8)
                vtk_colors = numpy_support.numpy_to_vtk(color_data, deep=False)
                vtk_colors.SetName("Colors")
                self.point_polydata.GetPointData().SetScalars(vtk_colors)
            
            # Performance tracking
            current_time = time.time()
            elapsed = current_time - self.last_update
            self.frame_times.append(elapsed)
            if len(self.frame_times) > 30:
                self.frame_times.pop(0)
            
            self.update_count += 1
            if self.update_count % 30 == 0:
                avg_fps = 1.0 / (sum(self.frame_times) / len(self.frame_times))
                print(f"Point Cloud FPS: {avg_fps:.1f}, Points: {len(points)}")
            
            self.last_update = current_time
            
            # Mark as modified
            self.point_polydata.Modified()
            
        except Exception as e:
            print(f"Error updating point cloud: {str(e)}")

def visualization_process(data_queue):
    width, height = 320, 240
    vis = PointCloudVisualizer(width, height)
    last_render = time.time()
    frames_processed = 0
    
    def timer_callback(obj, event):
        nonlocal last_render, frames_processed
        try:
            while True:  # Process all available frames
                camera_data = data_queue.get_nowait()
                vis.update_point_cloud(camera_data)
                frames_processed += 1
                
                # Limit max renders to ~30 FPS
                current_time = time.time()
                if current_time - last_render >= 0.033:  # ~30 FPS
                    vis.window.Render()
                    last_render = current_time
                    if frames_processed > 0:
                        print(f"Processed {frames_processed} frames")
                        frames_processed = 0
                    break
                    
        except queue.Empty:
            # No more frames to process
            pass
        except Exception as e:
            print(f"Error in timer callback: {e}")
    
    print("\nStarting visualization. Use mouse to interact:")
    print("- Left mouse: Rotate")
    print("- Middle mouse: Pan")
    print("- Right mouse: Zoom")
    print("- Press 'q' to exit")
    
    # Initialize
    vis.interactor.Initialize()
    
    # Set up timer for more frequent updates
    vis.interactor.AddObserver('TimerEvent', timer_callback)
    vis.interactor.CreateRepeatingTimer(16)  # ~60 FPS timer for checking queue
    
    # Start interaction
    vis.window.Render()
    vis.interactor.Start()

def main():
    # Create communication queue
    data_queue = Queue(maxsize=2)  # Small queue to prevent memory buildup
    
    # Create and start simulation process
    sim_process = Process(target=SimulationProcess(data_queue).run)
    sim_process.start()
    
    # Start visualization in main process
    try:
        visualization_process(data_queue)
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        sim_process.terminate()
        sim_process.join()

if __name__ == "__main__":
    main()