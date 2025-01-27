#!/usr/bin/env python3

import os
import xacro
from pathlib import Path

def main():
    # Get the directory containing this script
    script_dir = Path(__file__).parent.absolute()
    
    # Input xacro file
    xacro_file = script_dir / "urdf" / "ur5.xacro"
    
    # Output URDF file
    urdf_file = script_dir / "ur5_generated.urdf"
    
    # Process xacro file
    doc = xacro.process_file(str(xacro_file))
    
    # Generate URDF content
    urdf_content = doc.toprettyxml(indent='  ')
    
    # Save to file
    with open(urdf_file, 'w') as f:
        f.write(urdf_content)
        
    print(f"Generated URDF file: {urdf_file}")

if __name__ == "__main__":
    main() 