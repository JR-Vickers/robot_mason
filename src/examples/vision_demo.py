"""
Demo script showing how to use the vision system in the stone carving simulator.
"""

import sys
from pathlib import Path
import numpy as np
import cv2
from time import sleep

# Add parent directory to path to import from src
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.simulation.core import SimulationCore

def main():
    # Initialize simulation
    sim = SimulationCore(gui=True, fps=60)
    sim.setup()
    
    # Load robot and create stone block
    robot_path = str(Path(__file__).parent.parent.parent / "robots/ur5/ur5_generated.urdf")
    sim.load_robot(robot_path, base_position=[0, 0, 0])
    sim.create_stone_block(position=(0.5, 0, 0.5))
    
    # Add cameras at different positions
    sim.add_camera(
        "front",
        position=[1.5, 0, 1.0],
        target=[0.5, 0, 0.5],
        width=640,
        height=480
    )
    
    sim.add_camera(
        "top",
        position=[0.5, -0.5, 2.0],  # Moved back and slightly to the side
        target=[0.5, 0, 0.5],
        width=640,
        height=480,
        near_val=0.05,  # Adjusted near and far values
        far_val=4.0
    )
    
    # Main loop
    try:
        while True:
            # Step simulation
            sim.step()
            
            # Capture from both cameras
            front_view = sim.get_camera_view("front")
            top_view = sim.get_camera_view("top")
            
            if front_view and top_view:
                # Show RGB and depth images
                cv2.imshow("Front RGB", front_view["rgb"])
                # Normalize depth for visualization
                front_depth = (front_view["depth"] - front_view["depth"].min()) / (front_view["depth"].max() - front_view["depth"].min())
                top_depth = (top_view["depth"] - top_view["depth"].min()) / (top_view["depth"].max() - top_view["depth"].min())
                
                cv2.imshow("Front Depth", front_depth)
                cv2.imshow("Top RGB", top_view["rgb"])
                cv2.imshow("Top Depth", top_depth)
                
                # Exit if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        cv2.destroyAllWindows()
        sim.close()

if __name__ == "__main__":
    main() 