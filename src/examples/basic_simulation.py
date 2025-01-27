"""
Basic example demonstrating the core simulation functionality.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from simulation.core import SimulationCore
import time
import pybullet as p
import pybullet_data

def main():
    # Create and setup simulation
    sim = SimulationCore(gui=True, fps=240)
    sim.setup()
    
    # Create stone block
    stone_id = sim.create_stone_block(
        position=(0.5, 0, 0.5),
        size=(0.2, 0.2, 0.2)
    )
    
    if stone_id is None:
        print("Failed to create stone block")
        return
        
    # Load KUKA robot (using PyBullet's included URDF)
    robot_id = sim.load_robot("kuka_iiwa/model.urdf")
    if robot_id is None:
        print("Failed to load robot")
        return
    
    print("Simulation running. Press Ctrl+C to exit.")
    try:
        while True:
            sim.step()
    except KeyboardInterrupt:
        print("\nExiting simulation...")
    finally:
        sim.close()

if __name__ == "__main__":
    main() 