"""
Logging system for the robotic stone carving simulator.
Provides structured logging with different levels and categories.
"""

import logging
import sys
import pybullet as p
from pathlib import Path
from datetime import datetime
from typing import Optional

class SimLogger:
    def __init__(self, log_dir: str = "logs", level: int = logging.INFO):
        """
        Initialize logging system.
        
        Args:
            log_dir: Directory to store log files
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger("robot_simulator")
        self.logger.setLevel(level)
        
        # Remove existing handlers
        self.logger.handlers = []
        
        # Create formatters
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_formatter = logging.Formatter(
            '%(levelname)s: %(message)s'
        )
        
        # File handler (with timestamp in filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_handler = logging.FileHandler(
            self.log_dir / f"simulation_{timestamp}.log"
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
    def debug(self, msg: str) -> None:
        """Log debug message."""
        self.logger.debug(msg)
        
    def info(self, msg: str) -> None:
        """Log info message."""
        self.logger.info(msg)
        
    def warning(self, msg: str) -> None:
        """Log warning message."""
        self.logger.warning(msg)
        
    def error(self, msg: str) -> None:
        """Log error message."""
        self.logger.error(msg)
        
    def critical(self, msg: str) -> None:
        """Log critical message."""
        self.logger.critical(msg)
        
    def log_joint_states(self, robot_id: int) -> None:
        """Log current joint positions and velocities."""
        num_joints = p.getNumJoints(robot_id)
        joint_states = []
        
        for joint_idx in range(num_joints):
            state = p.getJointState(robot_id, joint_idx)
            joint_info = p.getJointInfo(robot_id, joint_idx)
            joint_name = joint_info[1].decode('utf-8')
            joint_states.append(f"{joint_name}: pos={state[0]:.3f}, vel={state[1]:.3f}")
            
        self.debug("Joint States:\n" + "\n".join(joint_states))
        
    def log_collision_info(self, robot_id: int, stone_id: int) -> None:
        """Log collision information between robot and stone."""
        points = p.getContactPoints(bodyA=robot_id, bodyB=stone_id)
        if points:
            self.warning(f"Collision detected! Contact points: {len(points)}")
            for point in points:
                self.debug(f"Contact normal force: {point[9]:.2f}N") 