"""
Basic example demonstrating the core simulation functionality.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from simulation.core import SimulationCore
from simulation.ui import RobotUI
from simulation.logger import SimLogger
import time
import pybullet as p
import pybullet_data
import logging

def main():
    # Initialize logger
    logger = SimLogger(level=logging.DEBUG)
    logger.info("Starting simulation...")
    
    # Create and setup simulation
    sim = SimulationCore(gui=True, fps=240)
    sim.setup()
    logger.info("Simulation environment created")
    
    # Create stone block
    stone_id = sim.create_stone_block(
        position=(0.5, 0, 0.5),
        size=(0.2, 0.2, 0.2)
    )
    
    if stone_id is None:
        logger.error("Failed to create stone block")
        return
    logger.info("Stone block created")
        
    # Load KUKA robot (using PyBullet's included URDF)
    robot_id = sim.load_robot("kuka_iiwa/model.urdf")
    if robot_id is None:
        logger.error("Failed to load robot")
        return
    logger.info("Robot loaded successfully")
    
    # Create UI controls
    ui = RobotUI(robot_id)
    logger.info("UI controls initialized")
    
    print("Simulation running. Press Ctrl+C to exit.")
    try:
        while True:
            ui.update()  # Update joint positions from sliders
            sim.step()
            
            # Log joint states and collisions periodically (every 100 steps)
            if sim.step_counter % 100 == 0:
                logger.log_joint_states(robot_id)
                logger.log_collision_info(robot_id, stone_id)
                
    except KeyboardInterrupt:
        logger.info("\nExiting simulation...")
    finally:
        sim.close()

if __name__ == "__main__":
    main() 