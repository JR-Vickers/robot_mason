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