"""
Demo script showing basic robot control functionality.
"""

import time
import math
import sys
from pathlib import Path
import pybullet as p

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.simulation.core import SimulationCore

def main():
    # Initialize simulation
    sim = SimulationCore(gui=True)
    sim.setup()
    
    # Load UR5 robot from our robots directory
    robot_urdf = str(Path(project_root) / "robots" / "ur5" / "ur5_generated.urdf")
    sim.load_robot(robot_urdf, base_position=[0, 0, 0.1])  # Raise robot 10cm off ground
    
    if sim.robot_control is None:
        print("Failed to initialize robot control")
        return
        
    # Create stone block target
    stone_pos = (0.4, 0, 0.2)  # Moved closer and higher
    sim.create_stone_block(position=stone_pos, size=(0.1, 0.1, 0.1))  # Made block smaller
    
    print("\nDemonstrating robot control capabilities...")
    print("Controls:")
    print("  E: Emergency stop - immediately halts all robot motion")
    print("  R: Reset emergency stop - resumes robot operation")
    print("  Arrow keys: Rotate camera")
    print("  +/-: Zoom camera")
    print("  WASD: Move camera target")
    print("\nDemo sequence:")
    print("1. Moving to different poses")
    print("2. Testing collision detection")
    print("3. Showing joint control")
    print("\nPress Ctrl+C to exit")
    
    try:
        # Demo 1: Move to different poses
        poses = [
            ([0.4, 0.0, 0.4], [0, -math.pi, 0]),  # Above stone, end effector pointing down
            ([0.4, 0.0, 0.25], [0, -math.pi, 0]),  # Near stone
            ([0.2, -0.2, 0.3], [0, -math.pi/2, 0]), # Different position
            ([0.3, 0.2, 0.35], [0, -math.pi/2, 0]),  # Another position
        ]
        
        current_pose = 0
        steps_in_pose = 0
        
        while True:
            # Only move if not e-stopped
            if not sim.robot_control.is_estopped:
                if steps_in_pose == 0:
                    # Start moving to next pose
                    pos, orn = poses[current_pose]
                    print(f"\nMoving to position: {pos}")
                    sim.robot_control.move_to_pose(pos, orn)
                
                steps_in_pose += 1
                
                # Move to next pose after 100 steps
                if steps_in_pose >= 100:
                    steps_in_pose = 0
                    current_pose = (current_pose + 1) % len(poses)
                    
                    # Check for collisions after each move
                    if sim.robot_control.is_in_collision():
                        print("Warning: Robot in collision!")
            
            sim.step()
            
    except KeyboardInterrupt:
        print("\nExiting demo...")
    finally:
        sim.close()

if __name__ == "__main__":
    main() 