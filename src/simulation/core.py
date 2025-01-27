"""
Core simulation module for the robotic stone carving simulator.
Handles PyBullet initialization and basic scene management.
"""

import pybullet as p
import pybullet_data
import numpy as np
from pathlib import Path
import time
from typing import Tuple, Optional

class SimulationCore:
    def __init__(self, gui: bool = True, fps: int = 240):
        """
        Initialize the simulation environment.
        
        Args:
            gui: Whether to use GUI (True) or direct mode (False)
            fps: Target frames per second for the simulation
        """
        self.gui = gui
        self.fps = fps
        self.dt = 1.0 / fps
        self.physics_client = None
        self.robot_id = None
        self.stone_id = None
        # Camera parameters
        self.cam_distance = 2.0
        self.cam_yaw = 45
        self.cam_pitch = -30
        self.cam_target = [0, 0, 0]
        
    def setup(self) -> None:
        """Initialize PyBullet and set up the basic scene."""
        # Connect to PyBullet
        self.physics_client = p.connect(p.GUI if self.gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Basic scene setup
        p.setGravity(0, 0, -9.81)
        p.setRealTimeSimulation(0)  # We'll step manually for better control
        
        # Disable default keyboard shortcuts
        p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 0)
        
        # Load ground plane
        p.loadURDF("plane.urdf")
        
        # Set initial camera position
        self.reset_camera()
        
        # Configure keyboard controls
        if self.gui:
            print("\nCamera Controls:")
            print("  Arrow Keys: Rotate camera")
            print("  +/-: Zoom in/out")
            print("  WASD: Move camera target")
            print("  R: Reset camera")
            print("\nOr use trackpad:")
            print("  Two-finger scroll: Zoom")
            print("  Two-finger swipe: Pan")
            print("  SHIFT + Two-finger swipe: Rotate")
        
    def reset_camera(self) -> None:
        """Reset camera to default position."""
        self.cam_distance = 2.0
        self.cam_yaw = 45
        self.cam_pitch = -30
        self.cam_target = [0, 0, 0]
        self.update_camera()
        
    def update_camera(self) -> None:
        """Update camera position based on current parameters."""
        p.resetDebugVisualizerCamera(
            cameraDistance=self.cam_distance,
            cameraYaw=self.cam_yaw,
            cameraPitch=self.cam_pitch,
            cameraTargetPosition=self.cam_target
        )
        
    def handle_keyboard(self) -> None:
        """Handle keyboard input for camera control."""
        keys = p.getKeyboardEvents()
        
        # Camera rotation (arrow keys)
        if p.B3G_LEFT_ARROW in keys and keys[p.B3G_LEFT_ARROW] & p.KEY_IS_DOWN:
            self.cam_yaw += 2
        if p.B3G_RIGHT_ARROW in keys and keys[p.B3G_RIGHT_ARROW] & p.KEY_IS_DOWN:
            self.cam_yaw -= 2
        if p.B3G_UP_ARROW in keys and keys[p.B3G_UP_ARROW] & p.KEY_IS_DOWN:
            self.cam_pitch += 2
        if p.B3G_DOWN_ARROW in keys and keys[p.B3G_DOWN_ARROW] & p.KEY_IS_DOWN:
            self.cam_pitch -= 2
            
        # Zoom (+ and -)
        if ord('=') in keys and keys[ord('=')] & p.KEY_IS_DOWN:
            self.cam_distance = max(0.1, self.cam_distance - 0.1)
        if ord('-') in keys and keys[ord('-')] & p.KEY_IS_DOWN:
            self.cam_distance += 0.1
            
        # Camera target movement (WASD)
        if ord('w') in keys and keys[ord('w')] & p.KEY_IS_DOWN:
            self.cam_target[1] += 0.05
        if ord('s') in keys and keys[ord('s')] & p.KEY_IS_DOWN:
            self.cam_target[1] -= 0.05
        if ord('a') in keys and keys[ord('a')] & p.KEY_IS_DOWN:
            self.cam_target[0] -= 0.05
        if ord('d') in keys and keys[ord('d')] & p.KEY_IS_DOWN:
            self.cam_target[0] += 0.05
            
        # Reset camera (R)
        if ord('r') in keys and keys[ord('r')] & p.KEY_WAS_TRIGGERED:
            self.reset_camera()
            
        self.update_camera()
        
    def load_robot(self, urdf_path: str) -> Optional[int]:
        """
        Load the robot arm into the simulation.
        
        Args:
            urdf_path: Path to the robot's URDF file
            
        Returns:
            Robot ID if successful, None if failed
        """
        try:
            self.robot_id = p.loadURDF(
                urdf_path,
                basePosition=[0, 0, 0],
                useFixedBase=True
            )
            return self.robot_id
        except p.error as e:
            print(f"Failed to load robot: {e}")
            return None
            
    def create_stone_block(
        self,
        position: Tuple[float, float, float] = (0.5, 0, 0.5),
        size: Tuple[float, float, float] = (0.2, 0.2, 0.2)
    ) -> Optional[int]:
        """
        Create a simple stone block in the simulation.
        
        Args:
            position: (x, y, z) position of the block center
            size: (length, width, height) of the block
            
        Returns:
            Block ID if successful, None if failed
        """
        collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=np.array(size)/2)
        visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=np.array(size)/2, 
                                         rgbaColor=[0.7, 0.7, 0.7, 1])
        
        try:
            self.stone_id = p.createMultiBody(
                baseMass=1.0,
                baseCollisionShapeIndex=collision_shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=position
            )
            return self.stone_id
        except p.error as e:
            print(f"Failed to create stone block: {e}")
            return None
    
    def step(self) -> None:
        """Step the simulation forward one timestep."""
        if self.gui:
            self.handle_keyboard()
        p.stepSimulation()
        time.sleep(self.dt)  # Maintain real-time simulation
        
    def reset(self) -> None:
        """Reset the simulation to initial state."""
        if self.stone_id is not None:
            p.removeBody(self.stone_id)
            self.stone_id = None
        if self.robot_id is not None:
            p.removeBody(self.robot_id)
            self.robot_id = None
        self.setup()
        
    def close(self) -> None:
        """Clean up and disconnect from PyBullet."""
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
            self.physics_client = None 