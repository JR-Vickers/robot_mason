"""
Multiprocess architecture for simulation and visualization.
"""

import time
import queue
from multiprocessing import Process
import numpy as np
import pybullet as p
from pathlib import Path

from ..vision.point_cloud import PointCloudProcessor, PointCloudTracker
from .core import SimulationCore

class SimulationProcess:
    def __init__(self, data_queue):
        self.data_queue = data_queue
        self.fps_display_id = None
        
    def run(self):
        # Initialize simulation
        sim = SimulationCore(gui=True, fps=240)
        sim.setup()
        
        # Configure PyBullet for maximum performance
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)
        p.setRealTimeSimulation(0)
        
        # Set physics parameters
        p.setPhysicsEngineParameter(fixedTimeStep=1.0/240.0)
        p.setPhysicsEngineParameter(numSolverIterations=10)
        p.setPhysicsEngineParameter(numSubSteps=1)
        
        # Create FPS display
        self.fps_display_id = p.addUserDebugText(
            "Sim FPS: --",
            [0.3, -0.9, 0.8],
            textSize=1.5,
            textColorRGB=[1, 1, 1]
        )
        
        # Initialize camera and robot
        self.setup_camera(sim)
        robot_id = self.setup_robot(sim)
        
        # Run simulation loop
        self.run_simulation_loop(sim, robot_id)
        
    def setup_camera(self, sim):
        """Set up camera and get parameters."""
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
        self.intrinsics, _ = processor.get_camera_parameters(
            fov=fov,
            aspect=aspect,
            near=near,
            far=far,
            view_matrix=sim.cameras["front"].view_matrix,
            width=width,
            height=height
        )
        
    def setup_robot(self, sim):
        """Load and set up robot."""
        robot_path = str(Path(__file__).parent.parent.parent / "robots/ur5/ur5_generated.urdf")
        return sim.load_robot(robot_path, base_position=[0, 0, 0])
        
    def run_simulation_loop(self, sim, robot_id):
        """Main simulation loop."""
        # Define joint positions for continuous motion
        joint_positions = [
            [0, -1.0, 1.0, -1.57, -1.57, 0],
            [1.57, -0.5, 0.5, -0.5, -1.57, 0],
            [3.14, -1.0, 1.0, -1.57, -1.57, 0],
            [-1.57, -0.5, 0.5, -0.5, -1.57, 0],
        ]
        
        current_target = 0
        interpolation_steps = 60
        step_counter = 0
        frame_count = 0
        last_fps_time = time.time()
        
        # Capture initial state
        self.capture_target_state(sim, robot_id, joint_positions[0])
        
        try:
            while True:
                # Update robot motion
                current_pos = joint_positions[current_target]
                next_pos = joint_positions[(current_target + 1) % len(joint_positions)]
                
                alpha = step_counter / interpolation_steps
                interpolated_pos = [
                    current_pos[i] + (next_pos[i] - current_pos[i]) * alpha
                    for i in range(len(current_pos))
                ]
                
                # Control robot
                for i in range(len(interpolated_pos)):
                    p.setJointMotorControl2(
                        robot_id, i, p.POSITION_CONTROL,
                        targetPosition=interpolated_pos[i],
                        force=100,
                        maxVelocity=2.0
                    )
                
                # Step simulation
                for _ in range(4):
                    p.stepSimulation()
                
                # Update counters
                step_counter += 1
                if step_counter >= interpolation_steps:
                    step_counter = 0
                    current_target = (current_target + 1) % len(joint_positions)
                
                # Send camera data
                self.send_camera_data(sim)
                
                # Update FPS display
                frame_count += 1
                if frame_count % 60 == 0:
                    self.update_fps_display(frame_count, last_fps_time)
                    frame_count = 0
                    last_fps_time = time.time()
                    
        except KeyboardInterrupt:
            print("\nClosing simulation...")
        finally:
            sim.close()
            
    def capture_target_state(self, sim, robot_id, initial_position):
        """Capture initial target state."""
        print("Capturing target point cloud...")
        
        # Move to initial position
        for i in range(len(initial_position)):
            p.setJointMotorControl2(
                robot_id, i, p.POSITION_CONTROL,
                targetPosition=initial_position[i],
                force=100
            )
        
        # Wait for robot to settle
        for _ in range(30):
            p.stepSimulation()
        
        # Capture and send target view
        target_view = sim.get_camera_view("front")
        if target_view is not None:
            target_data = {
                'view_matrix': np.array(sim.cameras["front"].view_matrix),
                'intrinsics': self.intrinsics,
                'extrinsics': np.linalg.inv(np.array(sim.cameras["front"].view_matrix).reshape(4, 4)),
                'depth': target_view["depth"],
                'rgb': target_view["rgb"],
                'is_target': True
            }
            try:
                self.data_queue.put(target_data, timeout=0.001)
            except queue.Full:
                pass
                
    def send_camera_data(self, sim):
        """Capture and send current camera view."""
        view = sim.get_camera_view("front")
        if view is not None:
            camera_data = {
                'view_matrix': np.array(sim.cameras["front"].view_matrix),
                'intrinsics': self.intrinsics,
                'extrinsics': np.linalg.inv(np.array(sim.cameras["front"].view_matrix).reshape(4, 4)),
                'depth': view["depth"],
                'rgb': view["rgb"]
            }
            try:
                self.data_queue.put(camera_data, timeout=0.001)
            except queue.Full:
                pass
                
    def update_fps_display(self, frame_count, last_time):
        """Update FPS counter in PyBullet window."""
        current_time = time.time()
        elapsed = current_time - last_time
        sim_fps = frame_count / elapsed if elapsed > 0 else 0
        
        if self.fps_display_id is not None:
            p.removeUserDebugItem(self.fps_display_id)
            self.fps_display_id = p.addUserDebugText(
                f"Sim FPS: {sim_fps:.1f}",
                [0.3, -0.9, 0.8],
                textSize=1.5,
                textColorRGB=[1, 1, 1]
            )

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
                    camera_data = self.input_queue.get(timeout=0.1)
                    
                    # Generate point cloud
                    pcd = self.processor.depth_to_point_cloud(
                        camera_data['depth'],
                        camera_data['intrinsics'],
                        camera_data['extrinsics'],
                        camera_data['rgb']
                    )
                    
                    # Process point cloud data
                    points = np.asarray(pcd.points)
                    colors = np.asarray(pcd.colors) if pcd.has_colors() else None
                    
                    # Handle target or update current
                    if not self.target_received and 'is_target' in camera_data:
                        self.handle_target_cloud(points, colors)
                    else:
                        self.handle_current_cloud(points, colors)
                        
                except queue.Empty:
                    continue
                    
        except KeyboardInterrupt:
            print("\nPoint cloud processor shutting down...")
        except Exception as e:
            print(f"Error in point cloud processor: {str(e)}")
            raise
            
    def handle_target_cloud(self, points, colors):
        """Process target point cloud."""
        self.tracker.set_target(points, colors)
        self.target_received = True
        result = {
            'points': points,
            'colors': colors,
            'is_target': True
        }
        self.send_result(result)
        
    def handle_current_cloud(self, points, colors):
        """Process current point cloud."""
        self.tracker.update_current(points, colors)
        if self.target_received:
            points, colors = self.tracker.get_difference_cloud()
        
        result = {
            'points': points,
            'colors': colors,
            'completion': self.tracker.get_completion_percentage()
        }
        self.send_result(result)
        
    def send_result(self, result):
        """Send processed data to visualization process."""
        try:
            self.output_queue.put(result, timeout=0.001)
        except queue.Full:
            pass 