"""
Test of real-time point cloud visualization using VTK with multiprocessing.
"""

import sys
from pathlib import Path
import time
from multiprocessing import Process, Queue
import queue
import os
import numpy as np

# Add parent directory to path to import from src
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.simulation.process import SimulationProcess, PointCloudProcessorProcess
from src.vision.visualizer import PointCloudVisualizer
from src.simulation.monitor import ProcessMonitor
from src.vision.safety_monitor import create_default_safety_envelope, IntrusionDetector

def visualization_process(processed_queue):
    """Main visualization process."""
    width, height = 320, 240
    vis = PointCloudVisualizer(width, height)
    last_render = time.time()
    frames_processed = 0
    
    # Create safety envelope
    robot_base = np.array([0, 0, 0])  # Adjust based on your robot's position
    safety_envelope = create_default_safety_envelope(
        robot_base=robot_base,
        workspace_radius=1.0,  # 1 meter radius
        height=1.5  # 1.5 meters height
    )
    
    # Create intrusion detector
    detector = IntrusionDetector(safety_envelope)
    
    # Add safety envelope visualization
    vis.add_safety_envelope(safety_envelope.points)
    
    def timer_callback(obj, event):
        nonlocal last_render, frames_processed
        try:
            while True:  # Process all available frames
                cloud_data = processed_queue.get_nowait()
                
                # Check for intrusions
                has_intrusion, intrusion_points = detector.process_point_cloud(cloud_data)
                
                # Update visualization
                vis.update_point_cloud(cloud_data)
                if has_intrusion:
                    print(f"WARNING: Intrusion detected with {len(intrusion_points)} points!")
                    vis.highlight_intrusion_points(intrusion_points)
                
                frames_processed += 1
                
                # Limit max renders to ~30 FPS
                current_time = time.time()
                if current_time - last_render >= 0.033:  # ~30 FPS
                    vis.window.Render()
                    last_render = current_time
                    if frames_processed > 0:
                        print(f"Processed {frames_processed} frames")
                        frames_processed = 0
                    break
                    
        except queue.Empty:
            # No more frames to process
            pass
        except Exception as e:
            print(f"Error in timer callback: {e}")
    
    # Start visualization
    vis.start(timer_callback)

def main():
    # Create communication queues with larger buffers
    raw_queue = Queue(maxsize=10)
    processed_queue = Queue(maxsize=10)
    
    # Create process monitor
    monitor = ProcessMonitor()
    
    # Create and start simulation process
    sim_process = Process(target=SimulationProcess(raw_queue).run)
    sim_process.start()
    monitor.register_process("simulation", sim_process.pid)
    
    # Create and start point cloud processor
    width, height = 320, 240
    processor_process = Process(
        target=PointCloudProcessorProcess(raw_queue, processed_queue, width, height).run
    )
    processor_process.start()
    monitor.register_process("processor", processor_process.pid)
    
    # Optimize process settings
    monitor.optimize_processes()
    
    # Start visualization in main process
    try:
        monitor.register_process("main", os.getpid())
        visualization_process(processed_queue)
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        # Print final statistics
        monitor.print_stats()
        
        # Cleanup processes
        sim_process.terminate()
        processor_process.terminate()
        sim_process.join()
        processor_process.join()

if __name__ == "__main__":
    main()