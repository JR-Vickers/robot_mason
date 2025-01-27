# Robotic Stone Carving Simulator

A Python-based simulation environment for robotic stone carving, demonstrating path planning, computer vision, and robotic control systems.

## Overview

This project simulates a robotic arm performing stone carving operations, showcasing:
- Real-time physics simulation using PyBullet
- Computer vision for progress tracking
- Path planning and optimization
- Material removal simulation
- Interactive visualization and control

## Setup

1. Create and activate Python virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Unix/macOS
# or
.\venv\Scripts\activate  # On Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

- `src/`: Source code
  - `simulation/`: Core simulation modules
  - `control/`: Robot control system
  - `vision/`: Computer vision processing
  - `examples/`: Example scripts and demos
- `tests/`: Unit and integration tests
- `docs/`: Documentation
- `requirements.txt`: Python dependencies

## Running the Simulation

Basic example:
```bash
python src/examples/basic_simulation.py
```

### Controls

Camera controls:
- Arrow keys: Rotate camera
- +/-: Zoom in/out
- WASD: Move camera target
- R: Reset camera view

Trackpad controls:
- Two-finger scroll: Zoom
- Two-finger swipe: Pan
- SHIFT + Two-finger swipe: Rotate

## Development Status

This project is under active development. See `plan.md` for the detailed development roadmap.

## License

[Insert chosen license] 