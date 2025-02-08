import numpy as np
import pybullet as p
from typing import List, Tuple, Optional
import math

class RobotControl:
    def __init__(self, robot_id: int, num_joints: int):
        self.robot_id = robot_id
        self.num_joints = num_joints
        self.joint_limits = self._get_joint_limits()
        self.end_effector_index = num_joints - 1
        self.is_estopped = False
        self.last_joint_positions = None
        
    def _get_joint_limits(self) -> List[Tuple[float, float]]:
        """Get joint limits for all controllable joints"""
        limits = []
        for i in range(self.num_joints):
            joint_info = p.getJointInfo(self.robot_id, i)
            lower, upper = joint_info[8], joint_info[9]
            limits.append((lower, upper))
        return limits
    
    def emergency_stop(self) -> None:
        """Activate emergency stop - freezes robot in current position"""
        self.is_estopped = True
        # Store current joint positions
        self.last_joint_positions = [p.getJointState(self.robot_id, i)[0] 
                                   for i in range(self.num_joints)]
        # Set joints to velocity control with zero velocity
        for i in range(self.num_joints):
            p.setJointMotorControl2(
                self.robot_id,
                i,
                p.VELOCITY_CONTROL,
                targetVelocity=0,
                force=100.0
            )
        print("EMERGENCY STOP ACTIVATED - Robot motion halted")
        
    def reset_estop(self) -> None:
        """Reset emergency stop if safe to do so"""
        self.is_estopped = False
        print("Emergency stop reset - Robot control re-enabled")
        # Return to position control of last known positions
        if self.last_joint_positions:
            self.set_joint_angles(self.last_joint_positions)
    
    def forward_kinematics(self, joint_angles: List[float]) -> np.ndarray:
        """Calculate end-effector pose given joint angles"""
        if self.is_estopped:
            print("Warning: Robot is e-stopped, forward kinematics for information only")
            
        # Set joints to given angles
        for i, angle in enumerate(joint_angles):
            p.resetJointState(self.robot_id, i, angle)
        
        # Get link state for end effector
        state = p.getLinkState(self.robot_id, self.end_effector_index)
        position = state[0]
        orientation = state[1]
        
        return np.array(list(position) + list(orientation))
    
    def inverse_kinematics(self, target_pos: List[float], target_orn: Optional[List[float]] = None) -> List[float]:
        """Calculate joint angles to achieve target end-effector pose"""
        if self.is_estopped:
            print("Warning: Robot is e-stopped, inverse kinematics for information only")
            return self.last_joint_positions if self.last_joint_positions else [0] * self.num_joints
            
        if target_orn is None:
            # Use current orientation if none specified
            current_orn = p.getLinkState(self.robot_id, self.end_effector_index)[1]
            target_orn = current_orn
            
        joint_angles = p.calculateInverseKinematics(
            self.robot_id,
            self.end_effector_index,
            target_pos,
            target_orn
        )
        
        # Clamp to joint limits
        clamped_angles = []
        for angle, (lower, upper) in zip(joint_angles, self.joint_limits):
            clamped_angles.append(np.clip(angle, lower, upper))
            
        return clamped_angles
    
    def set_joint_angles(self, joint_angles: List[float], max_force: float = 100.0) -> None:
        """Set robot joints to target angles"""
        if self.is_estopped:
            print("Cannot move robot: Emergency stop is active")
            return
            
        for i, angle in enumerate(joint_angles):
            p.setJointMotorControl2(
                self.robot_id,
                i,
                p.POSITION_CONTROL,
                targetPosition=angle,
                force=max_force
            )
            
    def move_to_pose(self, target_pos: List[float], target_orn: Optional[List[float]] = None,
                     max_force: float = 100.0) -> None:
        """Move end-effector to target pose"""
        if self.is_estopped:
            print("Cannot move robot: Emergency stop is active")
            return
            
        joint_angles = self.inverse_kinematics(target_pos, target_orn)
        self.set_joint_angles(joint_angles, max_force)
        
    def is_in_collision(self) -> bool:
        """Check if robot is in collision with environment"""
        return len(p.getContactPoints(bodyA=self.robot_id)) > 0
    
    def attach_tool(self, tool_id: int) -> None:
        """Attach tool to robot end-effector"""
        if self.is_estopped:
            print("Cannot attach tool: Emergency stop is active")
            return
            
        cid = p.createConstraint(
            self.robot_id,
            self.end_effector_index,
            tool_id,
            -1,
            p.JOINT_FIXED,
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        )
        p.changeConstraint(cid, maxForce=50)

    def get_end_effector_pose(self, joint_angles: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get end effector position and orientation for given joint angles.
        
        Args:
            joint_angles: List of joint angles
            
        Returns:
            Tuple of (position, orientation) where position is xyz and orientation is quaternion
        """
        # Set joints to given angles temporarily
        current_angles = [p.getJointState(self.robot_id, i)[0] for i in range(self.num_joints)]
        for i, angle in enumerate(joint_angles):
            p.resetJointState(self.robot_id, i, angle)
        
        # Get end effector state
        state = p.getLinkState(self.robot_id, self.end_effector_index)
        position = np.array(state[0])
        orientation = np.array(state[1])
        
        # Restore original joint angles
        for i, angle in enumerate(current_angles):
            p.resetJointState(self.robot_id, i, angle)
        
        return position, orientation

    def get_tool_direction(self, orientation: np.ndarray) -> np.ndarray:
        """
        Get tool direction vector from orientation quaternion.
        
        Args:
            orientation: Quaternion orientation
            
        Returns:
            Unit vector indicating tool direction
        """
        # Convert quaternion to rotation matrix
        rot_matrix = p.getMatrixFromQuaternion(orientation)
        rot_matrix = np.array(rot_matrix).reshape(3, 3)
        
        # Tool direction is typically along the Z-axis
        tool_direction = rot_matrix @ np.array([0, 0, 1])
        return tool_direction / np.linalg.norm(tool_direction)

    def plan_orientation(self, start_pos: np.ndarray, end_pos: np.ndarray, 
                        surface_normal: Optional[np.ndarray] = None,
                        angle: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Plan start and end orientations for tool path.
        
        Args:
            start_pos: Start position
            end_pos: End position
            surface_normal: Optional surface normal for normal-aligned motion
            angle: Optional angle from vertical for angled motion
            
        Returns:
            Tuple of (start_orientation, end_orientation) as quaternions
        """
        if surface_normal is not None:
            # Align tool with surface normal
            direction = surface_normal / np.linalg.norm(surface_normal)
        elif angle is not None:
            # Create angled direction from vertical
            direction = np.array([np.sin(angle), 0, -np.cos(angle)])
        else:
            # Default to vertical orientation
            direction = np.array([0, 0, -1])
        
        # Create rotation matrix that aligns Z-axis with desired direction
        z_axis = direction
        y_axis = np.array([0, 1, 0])  # Choose arbitrary up direction
        x_axis = np.cross(y_axis, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        
        rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))
        orientation = p.getQuaternionFromMatrix(rotation_matrix.flatten().tolist())
        
        return orientation, orientation  # Same orientation for start and end 

    def is_valid_config(self, joint_angles: List[float]) -> bool:
        """
        Check if joint configuration is valid (within limits).
        
        Args:
            joint_angles: List of joint angles to check
            
        Returns:
            True if configuration is valid, False otherwise
        """
        if len(joint_angles) != self.num_joints:
            return False
        
        for angle, (lower, upper) in zip(joint_angles, self.joint_limits):
            if angle < lower or angle > upper:
                return False
            
        return True

    def get_joint_limits(self) -> List[Tuple[float, float]]:
        """Get joint limits for all joints"""
        return self.joint_limits.copy() 