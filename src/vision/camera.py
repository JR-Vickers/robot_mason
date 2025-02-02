import pybullet as p
import numpy as np
from typing import Tuple, Optional, List, Dict
import cv2

class Camera:
    def __init__(
        self,
        position: List[float],
        target: List[float],
        up_vector: List[float] = [0, 0, 1],
        fov: float = 60.0,
        aspect: float = 1.0,
        near_val: float = 0.1,
        far_val: float = 10.0,
        width: int = 640,
        height: int = 480
    ):
        self.position = position
        self.target = target
        self.up_vector = up_vector
        self.fov = fov
        self.aspect = aspect
        self.near_val = near_val
        self.far_val = far_val
        self.width = width
        self.height = height
        
        # Calculate view and projection matrices
        self.view_matrix = p.computeViewMatrix(
            cameraEyePosition=position,
            cameraTargetPosition=target,
            cameraUpVector=up_vector
        )
        
        self.proj_matrix = p.computeProjectionMatrixFOV(
            fov=fov,
            aspect=aspect,
            nearVal=near_val,
            farVal=far_val
        )
        
    def capture(self) -> Dict[str, np.ndarray]:
        """Capture RGB and depth images from the camera."""
        # Get the image from PyBullet
        width, height, rgb_img, depth_img, seg_img = p.getCameraImage(
            width=self.width,
            height=self.height,
            viewMatrix=self.view_matrix,
            projectionMatrix=self.proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        
        # Convert the RGB image to proper format
        rgb = np.array(rgb_img, dtype=np.uint8).reshape(height, width, 4)
        rgb = rgb[:, :, :3]  # Remove alpha channel
        
        # Convert depth buffer to actual depth
        depth = np.array(depth_img).reshape(height, width)
        depth = self.far_val * self.near_val / (self.far_val - (self.far_val - self.near_val) * depth)
        
        return {
            "rgb": rgb,
            "depth": depth,
            "segmentation": np.array(seg_img).reshape(height, width)
        }
        
    def update_position(self, position: List[float], target: List[float]) -> None:
        """Update camera position and target."""
        self.position = position
        self.target = target
        self.view_matrix = p.computeViewMatrix(
            cameraEyePosition=position,
            cameraTargetPosition=target,
            cameraUpVector=self.up_vector
        ) 