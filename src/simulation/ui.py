"""
UI controls for the robotic stone carving simulator.
Provides slider-based joint control and other interactive elements.
"""

import pybullet as p
from typing import Dict, List, Optional

class RobotUI:
    def __init__(self, robot_id: int):
        """
        Initialize UI controls for robot manipulation.
        
        Args:
            robot_id: PyBullet body ID of the robot
        """
        self.robot_id = robot_id
        self.joint_sliders: Dict[int, int] = {}  # Maps joint index to slider ID
        self._setup_joint_controls()
        
    def _setup_joint_controls(self) -> None:
        """Create sliders for each controllable joint."""
        num_joints = p.getNumJoints(self.robot_id)
        
        for joint_idx in range(num_joints):
            joint_info = p.getJointInfo(self.robot_id, joint_idx)
            if joint_info[2] == p.JOINT_REVOLUTE:  # Only add sliders for revolute joints
                joint_name = joint_info[1].decode('utf-8')
                lower_limit = joint_info[8]  # Joint lower limit
                upper_limit = joint_info[9]  # Joint upper limit
                
                # Create slider
                slider_id = p.addUserDebugParameter(
                    paramName=f"Joint {joint_name}",
                    rangeMin=lower_limit,
                    rangeMax=upper_limit,
                    startValue=0
                )
                self.joint_sliders[joint_idx] = slider_id
    
    def update(self) -> None:
        """Update robot joint positions based on slider values."""
        try:
            for joint_idx, slider_id in self.joint_sliders.items():
                try:
                    target_pos = p.readUserDebugParameter(slider_id)
                    p.setJointMotorControl2(
                        bodyIndex=self.robot_id,
                        jointIndex=joint_idx,
                        controlMode=p.POSITION_CONTROL,
                        targetPosition=target_pos
                    )
                except p.error:
                    # Skip this slider if there's an error reading it
                    continue
        except Exception as e:
            # If we can't update the UI at all, just return silently
            # This happens when the simulation window is closed
            return 