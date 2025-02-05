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
from src.vision.point_cloud import PointCloudTracker, PointCloudProcessor

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
        self.fps_display_id = None  # Track debug text item
        
    def run(self):
        # Initialize simulation without GUI
        sim = SimulationCore(gui=True, fps=240)  # Increased FPS
        sim.setup()
        
        # Configure PyBullet for maximum performance
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)  # Disable GUI panels
        p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)
        p.setRealTimeSimulation(0)  # Disable real-time simulation
        
        # Set physics parameters for faster simulation
        p.setPhysicsEngineParameter(fixedTimeStep=1.0/240.0)  # Smaller timestep
        p.setPhysicsEngineParameter(numSolverIterations=10)
        p.setPhysicsEngineParameter(numSubSteps=1)
        
        # Load robot
        robot_path = str(Path(__file__).parent.parent.parent / "robots/ur5/ur5_generated.urdf")
        robot_id = sim.load_robot(robot_path, base_position=[0, 0, 0])
        
        # Add camera
        width, height = 320, 240
        fov = 60.0
        aspect = width / height
        near = 0.05
        far = 4.0
        
        sim.add_camera(
            "front",
            position=[1.5, 0, 1.0],
            target=[0.5, 0, 0.5],
            width=width,
            height=height,
            near_val=near,
            far_val=far
        )
        
        # Get camera parameters
        processor = PointCloudProcessor(width, height)
        intrinsics, _ = processor.get_camera_parameters(
            fov=fov,
            aspect=aspect,
            near=near,
            far=far,
            view_matrix=sim.cameras["front"].view_matrix,
            width=width,
            height=height
        )
        
        # Define joint positions for continuous motion
        joint_positions = [
            [0, -1.0, 1.0, -1.57, -1.57, 0],  # Position 1
            [1.57, -0.5, 0.5, -0.5, -1.57, 0],  # Position 2
            [3.14, -1.0, 1.0, -1.57, -1.57, 0],  # Position 3
            [-1.57, -0.5, 0.5, -0.5, -1.57, 0],  # Position 4
        ]
        current_target = 0
        interpolation_steps = 60  # Reduced steps for faster motion
        step_counter = 0
        
        print("Simulation running...")
        frame_count = 0
        last_fps_time = time.time()
        
        # Create FPS display in top-left corner
        self.fps_display_id = p.addUserDebugText(
            "Sim FPS: --",
            [0.3, -0.9, 0.8],  # Position in 3D space
            textSize=1.5,
            textColorRGB=[1, 1, 1]  # White text
        )
        
        # Capture initial state as target
        print("Capturing target point cloud...")
        # Move to initial position
        for i in range(len(joint_positions[0])):
            p.setJointMotorControl2(
                robot_id, i, p.POSITION_CONTROL,
                targetPosition=joint_positions[0][i],
                force=100
            )
        
        # Wait for robot to settle
        for _ in range(30):  # Reduced settling time
            p.stepSimulation()
        
        # Capture target view
        target_view = sim.get_camera_view("front")
        if target_view is not None:
            # Pack target data
            target_data = {
                'view_matrix': np.array(sim.cameras["front"].view_matrix),
                'intrinsics': intrinsics,
                'extrinsics': np.linalg.inv(np.array(sim.cameras["front"].view_matrix).reshape(4, 4)),
                'depth': target_view["depth"],
                'rgb': target_view["rgb"],
                'is_target': True
            }
            try:
                self.data_queue.put(target_data, timeout=0.001)
            except queue.Full:
                pass
        
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
                        force=100,
                        maxVelocity=2.0  # Increased velocity limit
                    )
                
                # Step simulation multiple times per frame for stability
                for _ in range(4):  # Reduced substeps
                    p.stepSimulation()
                
                # Update counters
                step_counter += 1
                if step_counter >= interpolation_steps:
                    step_counter = 0
                    current_target = (current_target + 1) % len(joint_positions)
                
                # Capture and send camera view
                view = sim.get_camera_view("front")
                if view is not None:
                    camera_data = {
                        'view_matrix': np.array(sim.cameras["front"].view_matrix),
                        'intrinsics': intrinsics,
                        'extrinsics': np.linalg.inv(np.array(sim.cameras["front"].view_matrix).reshape(4, 4)),
                        'depth': view["depth"],
                        'rgb': view["rgb"]
                    }
                    try:
                        self.data_queue.put(camera_data, timeout=0.001)
                    except queue.Full:
                        pass
                
                # FPS tracking
                frame_count += 1
                if frame_count % 60 == 0:
                    current_time = time.time()
                    elapsed = current_time - last_fps_time
                    sim_fps = 60 / elapsed if elapsed > 0 else 0
                    print(f"Simulation FPS: {sim_fps:.1f}")
                    
                    # Update FPS display in GUI
                    if self.fps_display_id is not None:
                        p.removeUserDebugItem(self.fps_display_id)
                        self.fps_display_id = p.addUserDebugText(
                            f"Sim FPS: {sim_fps:.1f}",
                            [0.3, -0.9, 0.8],
                            textSize=1.5,
                            textColorRGB=[1, 1, 1]
                        )
                    
                    last_fps_time = current_time
                
        except KeyboardInterrupt:
            print("\nClosing simulation...")
        finally:
            sim.close()

class PointCloudProcessorProcess:
    def __init__(self, input_queue, output_queue, width, height):
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.processor = PointCloudProcessor(width, height)
        self.tracker = PointCloudTracker()
        self.target_received = False
        
    def run(self):
        print("Point cloud processor running...")
        try:
            while True:
                try:
                    # Get camera data with timeout to allow for clean shutdown
                    camera_data = self.input_queue.get(timeout=0.1)
                    
                    # Generate point cloud
                    pcd = self.processor.depth_to_point_cloud(
                        camera_data['depth'],
                        camera_data['intrinsics'],
                        camera_data['extrinsics'],
                        camera_data['rgb']
                    )
                    
                    # Convert to numpy arrays
                    points = np.asarray(pcd.points)
                    colors = np.asarray(pcd.colors) if pcd.has_colors() else None
                    
                    # Handle target point cloud
                    if not self.target_received and 'is_target' in camera_data:
                        self.tracker.set_target(points, colors)
                        self.target_received = True
                        result = {
                            'points': points,
                            'colors': colors,
                            'is_target': True
                        }
                    else:
                        # Update tracker and get difference visualization
                        self.tracker.update_current(points, colors)
                        if self.target_received:
                            points, colors = self.tracker.get_difference_cloud()
                        
                        # Pack results
                        result = {
                            'points': points,
                            'colors': colors,
                            'completion': self.tracker.get_completion_percentage()
                        }
                    
                    # Send results to visualization process
                    try:
                        self.output_queue.put(result, timeout=0.001)
                    except queue.Full:
                        # Skip frame if queue is full
                        pass
                        
                except queue.Empty:
                    # No data available, continue waiting
                    continue
                    
        except KeyboardInterrupt:
            print("\nPoint cloud processor shutting down...")
        except Exception as e:
            print(f"Error in point cloud processor: {str(e)}")
            raise

class PointCloudVisualizer:
    def __init__(self, width, height):
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

def visualization_process(processed_queue):
    width, height = 320, 240
    vis = PointCloudVisualizer(width, height)
    last_render = time.time()
    frames_processed = 0
    
    def timer_callback(obj, event):
        nonlocal last_render, frames_processed
        try:
            while True:  # Process all available frames
                cloud_data = processed_queue.get_nowait()
                vis.update_point_cloud(cloud_data)
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
    # Create communication queues with larger buffers
    raw_queue = Queue(maxsize=10)  # Increased buffer size
    processed_queue = Queue(maxsize=10)  # Increased buffer size
    
    # Create and start simulation process with high priority
    sim_process = Process(target=SimulationProcess(raw_queue).run)
    sim_process.start()
    
    # Create and start point cloud processor on a different core
    width, height = 320, 240
    processor_process = Process(
        target=PointCloudProcessorProcess(raw_queue, processed_queue, width, height).run
    )
    
    # Set process priorities and affinities if on Unix-like system
    try:
        import os
        import psutil
        
        # Set simulation process to high priority and first core
        sim_pid = sim_process.pid
        sim_p = psutil.Process(sim_pid)
        sim_p.nice(10)  # Lower nice value = higher priority
        
        # Set processor to high priority and second core
        processor_process.start()
        proc_pid = processor_process.pid
        proc_p = psutil.Process(proc_pid)
        proc_p.nice(10)
        
        # Get the number of CPU cores
        cpu_count = os.cpu_count()
        if cpu_count >= 2:
            # Assign processes to different cores
            os.sched_setaffinity(sim_pid, {0})  # First core
            os.sched_setaffinity(proc_pid, {1})  # Second core
            
    except (ImportError, AttributeError):
        # Fall back to normal process start if on Windows or if psutil not available
        processor_process.start()
    
    # Start visualization in main process
    try:
        visualization_process(processed_queue)
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        sim_process.terminate()
        processor_process.terminate()
        sim_process.join()
        processor_process.join()

if __name__ == "__main__":
    main()